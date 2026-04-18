#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared training/evaluation utilities for Alzheimer_MRI OriginalDataset experiments.

This dataset does not provide train/test folders, so we build stratified splits from
the raw class folders under Alzheimer_MRI/OriginalDataset.

Important:
- class order is fixed to disease severity for ordinal losses
- training/logging style follows chest-x-ray-image_Loss
"""

import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
CHEST_LOSS_DIR = PROJECT_ROOT / "chest-x-ray-image_Loss"
for path in (PROJECT_ROOT, CHEST_LOSS_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chest_xray_loss_experiment_common import (  # noqa: E402
    DEFAULT_LOSS_ORDER as CHEST_SUPPORTED_LOSS_ORDER,
    Bottleneck,
    DualLogger,
    EarlyStopping,
    ResNet50WithInsertedModule,
    build_optimizer_with_groups,
    count_parameters,
    create_medical_loss as base_create_medical_loss,
    epoch_time,
    evaluate,
    get_pretrained_resnet50_state,
    has_trainable_parameters,
    load_checkpoint_states,
    load_pretrained_resnet50_backbone,
    set_seed,
    train_one_epoch,
)
from medical_losses import DistanceAwareSoftTargetLoss  # noqa: E402


SEED = 1234
DEFAULT_TOPK = (1, 2, 3)
DEFAULT_LOSS_ORDER = ("ce", "dast")
SUPPORTED_LOSS_ORDER = tuple(CHEST_SUPPORTED_LOSS_ORDER)
DEFAULT_CLASS_ORDER = (
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
)
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def float_slug(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def sanitize_run_tag(run_tag: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in run_tag.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-._")


class PathLabelDataset(data.Dataset):
    def __init__(self, samples, class_names, transform=None):
        self.samples = list(samples)
        self.class_names = list(class_names)
        self.transform = transform
        self.targets = [label for _, label in self.samples]
        self.classes = list(class_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@dataclass
class DataBundle:
    train_loader: data.DataLoader
    valid_loader: data.DataLoader
    test_loader: data.DataLoader
    train_targets: np.ndarray
    class_names: List[str]
    num_classes: int
    split_sizes: Dict[str, int]
    split_class_counts: Dict[str, List[int]]


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTENSIONS


def _class_counts_from_samples(samples, num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for _, label in samples:
        counts[int(label)] += 1
    return counts


def collect_ordered_samples(
    data_root: Path,
    class_order: Sequence[str],
) -> Tuple[List[Tuple[Path, int]], List[str]]:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    missing = [name for name in class_order if not (data_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing required class folders under {data_root}: {missing}"
        )

    samples: List[Tuple[Path, int]] = []
    for label_idx, class_name in enumerate(class_order):
        class_dir = data_root / class_name
        class_files = sorted(path for path in class_dir.iterdir() if _is_image_file(path))
        if not class_files:
            raise RuntimeError(f"No image files found under class folder: {class_dir}")
        samples.extend((path, label_idx) for path in class_files)

    if not samples:
        raise RuntimeError(f"No image files found under dataset root: {data_root}")

    return samples, list(class_order)


def build_alzheimer_mri_dataloaders(
    data_root: Path,
    test_ratio: float,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    image_size: int,
    seed: int,
    class_order: Sequence[str] = DEFAULT_CLASS_ORDER,
) -> DataBundle:
    samples, class_names = collect_ordered_samples(data_root, class_order)
    num_classes = len(class_names)
    all_targets = [label for _, label in samples]

    train_valid_samples, test_samples = train_test_split(
        samples,
        test_size=test_ratio,
        random_state=seed,
        stratify=all_targets,
    )
    train_valid_targets = [label for _, label in train_valid_samples]
    train_samples, valid_samples = train_test_split(
        train_valid_samples,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_valid_targets,
    )

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = PathLabelDataset(train_samples, class_names, transform=train_transform)
    valid_dataset = PathLabelDataset(valid_samples, class_names, transform=eval_transform)
    test_dataset = PathLabelDataset(test_samples, class_names, transform=eval_transform)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return DataBundle(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_targets=np.array([label for _, label in train_samples], dtype=np.int64),
        class_names=list(class_names),
        num_classes=num_classes,
        split_sizes={
            "train": len(train_dataset),
            "valid": len(valid_dataset),
            "test": len(test_dataset),
        },
        split_class_counts={
            "train": _class_counts_from_samples(train_samples, num_classes),
            "valid": _class_counts_from_samples(valid_samples, num_classes),
            "test": _class_counts_from_samples(test_samples, num_classes),
        },
    )


def resolve_losses_to_run() -> List[str]:
    env_losses = os.getenv("ALZHEIMER_LOSSES", "").strip()
    if not env_losses:
        return list(DEFAULT_LOSS_ORDER)

    requested = [x.strip().lower() for x in env_losses.split(",") if x.strip()]
    invalid = [x for x in requested if x not in SUPPORTED_LOSS_ORDER]
    if invalid:
        raise ValueError(f"Invalid loss names in ALZHEIMER_LOSSES: {invalid}")
    return requested


def create_alzheimer_medical_loss(
    loss_name: str,
    num_classes: int,
    class_counts: List[int],
    feat_dim: int,
    device: torch.device,
    dast_tau: float,
    dast_gamma: float,
) -> nn.Module:
    if loss_name.lower() == "dast":
        return DistanceAwareSoftTargetLoss(
            num_classes=num_classes,
            tau=dast_tau,
            gamma=dast_gamma,
        ).to(device)
    return base_create_medical_loss(
        loss_name=loss_name,
        num_classes=num_classes,
        class_counts=class_counts,
        feat_dim=feat_dim,
        device=device,
    )


def run_alzheimer_mri_medical_losses_experiments(
    script_stem: str,
    model_builder: Callable[[int], nn.Module],
    optimizer_group_divisors: Sequence[Tuple[str, float]],
    module_name: str,
    insert_after: str,
) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(
        os.getenv(
            "ALZHEIMER_DATA_ROOT",
            str(PROJECT_ROOT / "Alzheimer_MRI" / "OriginalDataset"),
        )
    )
    test_ratio = float(os.getenv("ALZHEIMER_TEST_RATIO", "0.2"))
    val_ratio = float(os.getenv("ALZHEIMER_VAL_RATIO", "0.1"))
    batch_size = int(os.getenv("ALZHEIMER_BATCH_SIZE", "32"))
    epochs = int(os.getenv("ALZHEIMER_EPOCHS", "60"))
    num_workers = int(os.getenv("ALZHEIMER_NUM_WORKERS", "2"))
    image_size = int(os.getenv("ALZHEIMER_IMAGE_SIZE", "224"))
    base_lr = float(os.getenv("ALZHEIMER_BASE_LR", "1e-4"))
    patience = int(os.getenv("ALZHEIMER_PATIENCE", "15"))
    early_delta = float(os.getenv("ALZHEIMER_EARLY_DELTA", "1e-4"))
    dast_tau = float(os.getenv("ALZHEIMER_DAST_TAU", "1.0"))
    dast_gamma = float(os.getenv("ALZHEIMER_DAST_GAMMA", "1.5"))
    run_tag = sanitize_run_tag(os.getenv("ALZHEIMER_RUN_TAG", ""))
    run_suffix = f"_{run_tag}" if run_tag else ""
    topk = DEFAULT_TOPK
    losses_to_run = resolve_losses_to_run()

    if dast_tau <= 0:
        raise ValueError("ALZHEIMER_DAST_TAU must be > 0.")
    if dast_gamma < 0:
        raise ValueError("ALZHEIMER_DAST_GAMMA must be >= 0.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = THIS_DIR / "logs"
    ckpt_dir = THIS_DIR / "checkpoints"
    log_path = logs_dir / f"{script_stem}{run_suffix}_{timestamp}.log"
    summary_path = logs_dir / f"{script_stem}{run_suffix}_{timestamp}_summary.csv"
    logger = DualLogger(log_path)
    log = logger.log

    try:
        log("=" * 90)
        log(f"Script: {script_stem}")
        if run_tag:
            log(f"Run tag: {run_tag}")
        log(f"Module: {module_name} | Insert after: {insert_after}")
        log(f"Device: {device}")
        if torch.cuda.is_available():
            log(f"CUDA: {torch.cuda.get_device_name(0)}")
        log(f"Data root: {data_root}")
        log(
            f"Class order: {list(DEFAULT_CLASS_ORDER)} "
            f"(used for labels/QWK/MAE/ordinal losses)"
        )
        log(
            f"Config | batch_size={batch_size}, epochs={epochs}, num_workers={num_workers}, "
            f"image_size={image_size}, base_lr={base_lr}, test_ratio={test_ratio}, "
            f"val_ratio_within_train={val_ratio}, patience={patience}"
        )
        log(f"Losses to run: {losses_to_run}")
        if "dast" in losses_to_run:
            log(f"DAST config | tau={dast_tau:.4f}, gamma={dast_gamma:.4f}")
        log("=" * 90)

        data_bundle = build_alzheimer_mri_dataloaders(
            data_root=data_root,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            seed=SEED,
        )

        log(
            f"Split sizes | train={data_bundle.split_sizes['train']}, "
            f"valid={data_bundle.split_sizes['valid']}, test={data_bundle.split_sizes['test']}"
        )
        log(f"Classes ({data_bundle.num_classes}): {data_bundle.class_names}")
        log(f"Train class counts: {data_bundle.split_class_counts['train']}")
        log(f"Valid class counts: {data_bundle.split_class_counts['valid']}")
        log(f"Test class counts : {data_bundle.split_class_counts['test']}")

        log("Loading torchvision ResNet50 pretrained weights once...")
        pretrained_state = get_pretrained_resnet50_state(data_bundle.num_classes)
        log(f"Pretrained state keys: {len(pretrained_state)}")

        summary_rows = []
        class_counts = data_bundle.split_class_counts["train"]

        for loss_name in losses_to_run:
            log("")
            log("#" * 90)
            log(f"Starting loss: {loss_name}")
            log("#" * 90)

            try:
                set_seed(SEED)
                model = model_builder(data_bundle.num_classes)
                loaded_n, total_n = load_pretrained_resnet50_backbone(model, pretrained_state)
                model = model.to(device)
                feat_dim = model.fc.in_features
                log(
                    f"Model params: {count_parameters(model):,} | "
                    f"pretrained loaded: {loaded_n}/{total_n} | feat_dim={feat_dim}"
                )

                criterion = create_alzheimer_medical_loss(
                    loss_name=loss_name,
                    num_classes=data_bundle.num_classes,
                    class_counts=class_counts,
                    feat_dim=feat_dim,
                    device=device,
                    dast_tau=dast_tau,
                    dast_gamma=dast_gamma,
                )
                log(f"Criterion: {criterion.__class__.__name__}")
                log(f"Criterion trainable params: {count_parameters(criterion):,}")
                if loss_name == "dast":
                    log(f"Criterion hparams | tau={dast_tau:.4f}, gamma={dast_gamma:.4f}")

                extra_modules = None
                if has_trainable_parameters(criterion):
                    extra_modules = [("criterion", criterion, 1.0)]

                optimizer, max_lrs = build_optimizer_with_groups(
                    model=model,
                    base_lr=base_lr,
                    group_divisors=optimizer_group_divisors,
                    extra_modules=extra_modules,
                )
                scheduler = lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lrs,
                    total_steps=epochs * len(data_bundle.train_loader),
                )

                best_path = ckpt_dir / f"best_{script_stem}_{loss_name}{run_suffix}.pt"
                early_stopping = EarlyStopping(
                    patience=patience,
                    delta=early_delta,
                    save_path=best_path,
                )

                best_val_macro = -1.0
                best_valid_loss = float("inf")
                best_epoch = 0
                trained_epochs = 0

                for epoch in range(epochs):
                    start_t = time.time()
                    train_loss, train_top = train_one_epoch(
                        model=model,
                        loader=data_bundle.train_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        scheduler=scheduler,
                        device=device,
                        loss_name=loss_name,
                        topk=topk,
                    )
                    valid_loss, valid_top, valid_metrics = evaluate(
                        model=model,
                        loader=data_bundle.valid_loader,
                        criterion=criterion,
                        device=device,
                        loss_name=loss_name,
                        num_classes=data_bundle.num_classes,
                        class_names=data_bundle.class_names,
                        topk=topk,
                    )
                    end_t = time.time()
                    mins, secs = epoch_time(start_t, end_t)
                    trained_epochs = epoch + 1

                    log(f"Epoch {epoch + 1:02d}/{epochs} | Time {mins}m{secs}s")
                    log(
                        f"  Train | loss={train_loss:.4f} | top1={train_top['top1'] * 100:.2f}% "
                        f"| top2={train_top['top2'] * 100:.2f}% | top3={train_top['top3'] * 100:.2f}%"
                    )
                    log(
                        f"  Valid | loss={valid_loss:.4f} | top1={valid_top['top1'] * 100:.2f}% "
                        f"| top2={valid_top['top2'] * 100:.2f}% | top3={valid_top['top3'] * 100:.2f}% "
                        f"| acc={valid_metrics['acc'] * 100:.2f}% | bal_acc={valid_metrics['balanced_acc'] * 100:.2f}% "
                        f"| macro_f1={valid_metrics['macro_f1']:.4f} | weighted_f1={valid_metrics['weighted_f1']:.4f} "
                        f"| precision_macro={valid_metrics['precision_macro']:.4f} | recall_macro={valid_metrics['recall_macro']:.4f} "
                        f"| qwk={valid_metrics['qwk']:.4f} | mae={valid_metrics['mae']:.4f}"
                    )
                    if valid_metrics["ovr_roc_auc_macro"] is not None:
                        log(
                            f"        ovr_roc_auc_macro={valid_metrics['ovr_roc_auc_macro']:.4f} "
                            f"| ovr_pr_auc_macro={valid_metrics['ovr_pr_auc_macro']:.4f}"
                        )

                    improved = early_stopping(valid_metrics["macro_f1"], model, criterion)
                    if improved:
                        best_val_macro = valid_metrics["macro_f1"]
                        best_valid_loss = valid_loss
                        best_epoch = epoch + 1
                        log(f"  -> best macro_f1 updated to {best_val_macro:.4f} (epoch {best_epoch})")

                    if early_stopping.early_stop:
                        log(f"  -> early stopping triggered at epoch {epoch + 1}")
                        break

                if best_path.exists():
                    load_checkpoint_states(best_path, model, device, criterion)

                test_loss, test_top, test_metrics = evaluate(
                    model=model,
                    loader=data_bundle.test_loader,
                    criterion=criterion,
                    device=device,
                    loss_name=loss_name,
                    num_classes=data_bundle.num_classes,
                    class_names=data_bundle.class_names,
                    topk=topk,
                )

                log("")
                log(f"[TEST] loss={loss_name}")
                log(
                    f"  Test | loss={test_loss:.4f} | top1={test_top['top1'] * 100:.2f}% "
                    f"| top2={test_top['top2'] * 100:.2f}% | top3={test_top['top3'] * 100:.2f}% "
                    f"| acc={test_metrics['acc'] * 100:.2f}% | bal_acc={test_metrics['balanced_acc'] * 100:.2f}% "
                    f"| macro_f1={test_metrics['macro_f1']:.4f} | weighted_f1={test_metrics['weighted_f1']:.4f} "
                    f"| precision_macro={test_metrics['precision_macro']:.4f} | recall_macro={test_metrics['recall_macro']:.4f} "
                    f"| qwk={test_metrics['qwk']:.4f} | mae={test_metrics['mae']:.4f}"
                )
                if test_metrics["ovr_roc_auc_macro"] is not None:
                    log(
                        f"        ovr_roc_auc_macro={test_metrics['ovr_roc_auc_macro']:.4f} "
                        f"| ovr_pr_auc_macro={test_metrics['ovr_pr_auc_macro']:.4f}"
                    )
                log("  Confusion Matrix:")
                log(str(test_metrics["confusion_matrix"]))
                log("  Classification Report:")
                log(str(test_metrics["classification_report"]))

                summary_rows.append({
                    "run_tag": run_tag,
                    "loss_name": loss_name,
                    "status": "success",
                    "dast_tau": dast_tau if loss_name == "dast" else None,
                    "dast_gamma": dast_gamma if loss_name == "dast" else None,
                    "trained_epochs": trained_epochs,
                    "best_epoch": best_epoch,
                    "best_valid_macro_f1": best_val_macro,
                    "best_valid_loss": best_valid_loss,
                    "test_loss": test_loss,
                    "test_top1": test_top["top1"],
                    "test_top2": test_top["top2"],
                    "test_top3": test_top["top3"],
                    "test_acc": test_metrics["acc"],
                    "test_balanced_acc": test_metrics["balanced_acc"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_weighted_f1": test_metrics["weighted_f1"],
                    "test_precision_macro": test_metrics["precision_macro"],
                    "test_recall_macro": test_metrics["recall_macro"],
                    "test_qwk": test_metrics["qwk"],
                    "test_mae": test_metrics["mae"],
                    "test_ovr_roc_auc_macro": test_metrics["ovr_roc_auc_macro"],
                    "test_ovr_pr_auc_macro": test_metrics["ovr_pr_auc_macro"],
                    "checkpoint_path": str(best_path),
                })
            except Exception as loss_exc:
                log(f"[ERROR] loss={loss_name} failed: {loss_exc}")
                log(traceback.format_exc())
                summary_rows.append({
                    "run_tag": run_tag,
                    "loss_name": loss_name,
                    "status": "failed",
                    "dast_tau": dast_tau if loss_name == "dast" else None,
                    "dast_gamma": dast_gamma if loss_name == "dast" else None,
                    "error": str(loss_exc),
                })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        log("")
        log("=" * 90)
        log(f"Summary CSV saved: {summary_path}")
        if not summary_df.empty and "status" in summary_df.columns:
            success_df = summary_df[summary_df["status"] == "success"].copy()
            if not success_df.empty and "test_macro_f1" in success_df.columns:
                success_df = success_df.sort_values("test_macro_f1", ascending=False)
                log("Top results by test_macro_f1:")
                for _, row in success_df.iterrows():
                    log(
                        f"  {row['loss_name']}: macro_f1={row['test_macro_f1']:.4f}, "
                        f"acc={row['test_acc'] * 100:.2f}%, bal_acc={row['test_balanced_acc'] * 100:.2f}%"
                    )
            else:
                log("No successful loss run found.")
        log(f"Log file saved: {log_path}")
        log("=" * 90)
    finally:
        logger.close()
