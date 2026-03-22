#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared training/evaluation utilities for HAM10000 + ResNet50 + attention modules.

This runner provides:
- unified metrics (APTOS-style + HAM10000-friendly)
- all medical losses in medical_losses.py (excluding plain nn.CrossEntropyLoss baseline)
- per-loss independent training with early stopping
- dedicated log file and summary CSV for each script
"""

import os
import sys
import time
import random
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
import torch.optim as optim
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
)


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_losses import (  # noqa: E402
    ClassBalancedFocalCELoss,
    OrdinalFocalMSELoss,
    SymmetricCrossEntropyLoss,
    GeneralizedCrossEntropyLoss,
    DistanceAwareSoftTargetLoss,
    PrototypeConsistencyOrdinalLoss,
    AdaptiveOrdinalMarginLoss,
)


SEED = 1234
DEFAULT_TOPK = (1, 3)
DEFAULT_LOSS_ORDER = (
    "cb_focal_ce",
    "ordinal_focal_mse",
    "sce",
    "gce",
    "dast",
    "pcol",
    "aom",
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DualLogger:
    def __init__(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path = file_path
        self.fp = open(file_path, "w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        text = str(message)
        print(text, flush=True)
        self.fp.write(text + "\n")
        self.fp.flush()

    def close(self) -> None:
        if not self.fp.closed:
            self.fp.close()


class EarlyStopping:
    def __init__(self, patience: int, delta: float, save_path: Path):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_score: Optional[float] = None
        self.num_bad_epochs = 0
        self.early_stop = False
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, score: float, model: nn.Module) -> bool:
        improved = False
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.num_bad_epochs = 0
            torch.save(model.state_dict(), self.save_path)
            improved = True
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.early_stop = True
        return improved


class ISICDataset(data.Dataset):
    """
    metadata_df must contain:
      - image_id
      - dx
      - image_dir
    """

    def __init__(self, metadata_df: pd.DataFrame, transform=None):
        self.df = metadata_df.reset_index(drop=True).copy()
        self.transform = transform

        self.labels = sorted(self.df["dx"].unique().tolist())
        self.label_to_idx = {lb: i for i, lb in enumerate(self.labels)}

        self.img_paths: List[str] = []
        self.targets: List[int] = []
        for _, row in self.df.iterrows():
            img_path = os.path.join(row["image_dir"], row["image_id"] + ".jpg")
            if os.path.exists(img_path):
                self.img_paths.append(img_path)
                self.targets.append(self.label_to_idx[row["dx"]])

        if not self.img_paths:
            raise RuntimeError("No valid images found in ISICDataset.")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        y = self.targets[idx]

        x = Image.open(img_path).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, y


class TransformSubset(data.Dataset):
    def __init__(self, base_dataset: ISICDataset, indices, transform=None):
        self.base = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        img_path = self.base.img_paths[base_idx]
        y = self.base.targets[base_idx]

        x = Image.open(img_path).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, y


@dataclass
class DataBundle:
    train_loader: data.DataLoader
    valid_loader: data.DataLoader
    test_loader: data.DataLoader
    train_targets: np.ndarray
    class_names: List[str]
    num_classes: int
    split_sizes: Dict[str, int]


def _build_valid_dataframe(data_dir: Path) -> pd.DataFrame:
    metadata_path = data_dir / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)
    part1_dir = data_dir / "HAM10000_images_part_1"
    part2_dir = data_dir / "HAM10000_images_part_2"
    image_dirs = [d for d in (part1_dir, part2_dir) if d.exists()]
    if not image_dirs:
        raise FileNotFoundError("No image directories found for HAM10000.")

    rows = []
    for _, row in metadata_df.iterrows():
        image_id = row["image_id"]
        found_dir = None
        for img_dir in image_dirs:
            if (img_dir / f"{image_id}.jpg").exists():
                found_dir = str(img_dir)
                break
        if found_dir is not None:
            r = row.copy()
            r["image_dir"] = found_dir
            rows.append(r)

    valid_df = pd.DataFrame(rows).reset_index(drop=True)
    if valid_df.empty:
        raise RuntimeError("No valid images found after checking disk paths.")
    return valid_df


def build_ham10000_dataloaders(
    batch_size: int,
    num_workers: int,
    image_size: int,
    seed: int,
    data_dir: Path,
) -> DataBundle:
    valid_df = _build_valid_dataframe(data_dir)
    class_names = sorted(valid_df["dx"].unique().tolist())
    num_classes = len(class_names)

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(image_size, padding=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_dataset = ISICDataset(valid_df, transform=None)
    targets = np.array(base_dataset.targets)

    indices = np.arange(len(base_dataset))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=seed
    )
    train_targets = targets[train_idx]
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.1, stratify=train_targets, random_state=seed
    )

    train_dataset = TransformSubset(base_dataset, train_idx, transform=train_tf)
    valid_dataset = TransformSubset(base_dataset, valid_idx, transform=eval_tf)
    test_dataset = TransformSubset(base_dataset, test_idx, transform=eval_tf)

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
        train_targets=train_targets,
        class_names=class_names,
        num_classes=num_classes,
        split_sizes={
            "train": len(train_dataset),
            "valid": len(valid_dataset),
            "test": len(test_dataset),
        },
    )


def calculate_topk_accuracy(logits: torch.Tensor, y: torch.Tensor, ks=(1, 3)) -> Dict[str, float]:
    with torch.no_grad():
        num_classes = logits.size(1)
        ks = tuple(sorted(set(min(k, num_classes) for k in ks)))
        max_k = max(ks)
        _, top_pred = logits.topk(max_k, dim=1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))

        out = {}
        batch_size = y.size(0)
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0).item()
            out[f"top{k}"] = correct_k / batch_size
        return out


def compute_eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    class_names: Sequence[str],
) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    metrics["mae"] = float(np.mean(np.abs(y_true.astype(np.float32) - y_pred.astype(np.float32))))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=list(class_names), digits=4, zero_division=0
    )

    metrics["ovr_roc_auc_macro"] = None
    metrics["ovr_pr_auc_macro"] = None
    try:
        y_true_oh = np.eye(num_classes)[y_true]
        metrics["ovr_roc_auc_macro"] = roc_auc_score(
            y_true_oh, y_prob, average="macro", multi_class="ovr"
        )
        metrics["ovr_pr_auc_macro"] = average_precision_score(
            y_true_oh, y_prob, average="macro"
        )
    except Exception:
        pass

    return metrics


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    elapsed = end_time - start_time
    return int(elapsed // 60), int(elapsed % 60)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pretrained_resnet50_state(num_classes: int) -> Dict[str, torch.Tensor]:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        tv_model = models.resnet50(weights=weights)
    except Exception:
        try:
            tv_model = models.resnet50(weights="IMAGENET1K_V1")
        except Exception:
            tv_model = models.resnet50(pretrained=True)

    in_features = tv_model.fc.in_features
    tv_model.fc = nn.Linear(in_features, num_classes)
    return tv_model.state_dict()


def load_pretrained_resnet50_backbone(
    model: nn.Module,
    pretrained_state: Dict[str, torch.Tensor],
) -> Tuple[int, int]:
    model_dict = model.state_dict()
    filtered = {
        k: v for k, v in pretrained_state.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(filtered)
    model.load_state_dict(model_dict, strict=False)
    return len(filtered), len(model_dict)


def build_optimizer_with_groups(
    model: nn.Module,
    base_lr: float,
    group_divisors: Sequence[Tuple[str, float]],
) -> Tuple[optim.Optimizer, List[float]]:
    param_groups = []
    max_lrs: List[float] = []

    for attr_name, divisor in group_divisors:
        module = getattr(model, attr_name)
        params = list(module.parameters())
        if not params:
            continue
        lr = base_lr / float(divisor)
        param_groups.append({"params": params, "lr": lr})
        max_lrs.append(lr)

    optimizer = optim.Adam(param_groups, lr=base_lr)
    return optimizer, max_lrs


def create_medical_loss(
    loss_name: str,
    num_classes: int,
    class_counts: List[int],
    feat_dim: int,
    device: torch.device,
) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "cb_focal_ce":
        criterion = ClassBalancedFocalCELoss(
            class_counts=class_counts,
            beta=0.9999,
            gamma=2.0,
            smoothing=0.05,
        )
    elif loss_name == "ordinal_focal_mse":
        criterion = OrdinalFocalMSELoss(
            num_classes=num_classes,
            alpha_ce=1.0,
            alpha_mse=0.3,
            gamma=2.0,
        )
    elif loss_name == "sce":
        criterion = SymmetricCrossEntropyLoss(alpha=1.0, beta=0.5, num_classes=num_classes)
    elif loss_name == "gce":
        criterion = GeneralizedCrossEntropyLoss(q=0.7)
    elif loss_name == "dast":
        criterion = DistanceAwareSoftTargetLoss(num_classes=num_classes, tau=1.0, gamma=1.5)
    elif loss_name == "pcol":
        criterion = PrototypeConsistencyOrdinalLoss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            lambda_proto=0.2,
            lambda_order=0.05,
        )
    elif loss_name == "aom":
        criterion = AdaptiveOrdinalMarginLoss(
            num_classes=num_classes,
            margin_base=0.15,
            power=1.0,
        )
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    return criterion.to(device)


def _compute_loss(
    criterion: nn.Module,
    loss_name: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    if loss_name == "pcol":
        return criterion(logits, targets, features)
    return criterion(logits, targets)


def train_one_epoch(
    model: nn.Module,
    loader: data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scheduler: Optional[lr_scheduler._LRScheduler],
    device: torch.device,
    loss_name: str,
    topk=(1, 3),
) -> Tuple[float, Dict[str, float]]:
    model.train()
    epoch_loss = 0.0
    epoch_top = {f"top{k}": 0.0 for k in topk}

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, features = model(x)
        loss = _compute_loss(criterion, loss_name, logits, y, features)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
        batch_top = calculate_topk_accuracy(logits, y, ks=topk)
        for k, v in batch_top.items():
            epoch_top[k] += v

    epoch_loss /= len(loader)
    for k in epoch_top:
        epoch_top[k] /= len(loader)
    return epoch_loss, epoch_top


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    loss_name: str,
    num_classes: int,
    class_names: Sequence[str],
    topk=(1, 3),
) -> Tuple[float, Dict[str, float], Dict[str, object]]:
    model.eval()
    epoch_loss = 0.0
    epoch_top = {f"top{k}": 0.0 for k in topk}

    all_y = []
    all_pred = []
    all_prob = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits, features = model(x)
        loss = _compute_loss(criterion, loss_name, logits, y, features)
        epoch_loss += loss.item()

        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

        batch_top = calculate_topk_accuracy(logits, y, ks=topk)
        for k, v in batch_top.items():
            epoch_top[k] += v

        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

    epoch_loss /= len(loader)
    for k in epoch_top:
        epoch_top[k] /= len(loader)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    metrics = compute_eval_metrics(y_true, y_pred, y_prob, num_classes, class_names)
    return epoch_loss, epoch_top, metrics


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        self.relu = nn.ReLU(inplace=True)

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.relu(x + identity)
        return x


class ResNet50WithInsertedModule(nn.Module):
    def __init__(self, num_classes: int, inserted_module: nn.Module, insert_after: str):
        super().__init__()
        if insert_after not in {"layer2", "layer3"}:
            raise ValueError("insert_after must be 'layer2' or 'layer3'")

        self.insert_after = insert_after
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 3, 64, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 4, 128, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 6, 256, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 3, 512, stride=2)

        self.inserted_module = inserted_module
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _make_layer(self, block, n_blocks, channels, stride=1):
        layers = []
        downsample = (self.in_channels != block.expansion * channels) or (stride != 1)
        layers.append(block(self.in_channels, channels, stride=stride, downsample=downsample))
        self.in_channels = block.expansion * channels
        for _ in range(1, n_blocks):
            layers.append(block(self.in_channels, channels, stride=1, downsample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        if self.insert_after == "layer2":
            x = self.inserted_module(x)
        x = self.layer3(x)
        if self.insert_after == "layer3":
            x = self.inserted_module(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.size(0), -1)
        logits = self.fc(h)
        return logits, h


def resolve_losses_to_run() -> List[str]:
    env_losses = os.getenv("HAM10000_LOSSES", "").strip()
    if not env_losses:
        return list(DEFAULT_LOSS_ORDER)

    requested = [x.strip().lower() for x in env_losses.split(",") if x.strip()]
    invalid = [x for x in requested if x not in DEFAULT_LOSS_ORDER]
    if invalid:
        raise ValueError(f"Invalid loss names in HAM10000_LOSSES: {invalid}")
    return requested


def run_ham10000_medical_losses_experiments(
    script_stem: str,
    model_builder: Callable[[int], nn.Module],
    optimizer_group_divisors: Sequence[Tuple[str, float]],
    module_name: str,
    insert_after: str,
) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = int(os.getenv("HAM10000_BATCH_SIZE", "32"))
    epochs = int(os.getenv("HAM10000_EPOCHS", "40"))
    num_workers = int(os.getenv("HAM10000_NUM_WORKERS", "2"))
    image_size = int(os.getenv("HAM10000_IMAGE_SIZE", "224"))
    base_lr = float(os.getenv("HAM10000_BASE_LR", "1e-4"))
    patience = int(os.getenv("HAM10000_PATIENCE", "10"))
    early_delta = float(os.getenv("HAM10000_EARLY_DELTA", "1e-4"))
    topk = DEFAULT_TOPK
    data_dir = Path(os.getenv("HAM10000_DATA_DIR", str(PROJECT_ROOT / "ISIC")))

    losses_to_run = resolve_losses_to_run()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = THIS_DIR / "logs"
    ckpt_dir = THIS_DIR / "checkpoints"
    log_path = logs_dir / f"{script_stem}_{timestamp}.log"
    summary_path = logs_dir / f"{script_stem}_{timestamp}_summary.csv"
    logger = DualLogger(log_path)
    log = logger.log

    try:
        log("=" * 90)
        log(f"Script: {script_stem}")
        log(f"Module: {module_name} | Insert after: {insert_after}")
        log(f"Device: {device}")
        if torch.cuda.is_available():
            log(f"CUDA: {torch.cuda.get_device_name(0)}")
        log(f"Data dir: {data_dir}")
        log(
            f"Config | batch_size={batch_size}, epochs={epochs}, num_workers={num_workers}, "
            f"image_size={image_size}, base_lr={base_lr}, patience={patience}"
        )
        log(f"Losses to run: {losses_to_run}")
        log("=" * 90)

        data_bundle = build_ham10000_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            seed=SEED,
            data_dir=data_dir,
        )

        log(
            f"Split sizes | train={data_bundle.split_sizes['train']}, "
            f"valid={data_bundle.split_sizes['valid']}, test={data_bundle.split_sizes['test']}"
        )
        log(f"Classes ({data_bundle.num_classes}): {data_bundle.class_names}")

        class_counts = np.bincount(
            data_bundle.train_targets, minlength=data_bundle.num_classes
        ).tolist()
        log(f"Train class counts: {class_counts}")

        log("Loading torchvision ResNet50 pretrained weights once...")
        pretrained_state = get_pretrained_resnet50_state(data_bundle.num_classes)
        log(f"Pretrained state keys: {len(pretrained_state)}")

        summary_rows = []

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

                criterion = create_medical_loss(
                    loss_name=loss_name,
                    num_classes=data_bundle.num_classes,
                    class_counts=class_counts,
                    feat_dim=feat_dim,
                    device=device,
                )
                log(f"Criterion: {criterion.__class__.__name__}")

                optimizer, max_lrs = build_optimizer_with_groups(
                    model=model,
                    base_lr=base_lr,
                    group_divisors=optimizer_group_divisors,
                )
                total_steps = epochs * len(data_bundle.train_loader)
                scheduler = lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lrs,
                    total_steps=total_steps,
                )

                best_path = ckpt_dir / f"best_{script_stem}_{loss_name}.pt"
                early_stopping = EarlyStopping(
                    patience=patience,
                    delta=early_delta,
                    save_path=best_path,
                )

                best_val_macro = -1.0
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
                        f"| top3={train_top['top3'] * 100:.2f}%"
                    )
                    log(
                        f"  Valid | loss={valid_loss:.4f} | top1={valid_top['top1'] * 100:.2f}% "
                        f"| top3={valid_top['top3'] * 100:.2f}% | acc={valid_metrics['acc'] * 100:.2f}% "
                        f"| bal_acc={valid_metrics['balanced_acc'] * 100:.2f}% | macro_f1={valid_metrics['macro_f1']:.4f} "
                        f"| weighted_f1={valid_metrics['weighted_f1']:.4f} | precision_macro={valid_metrics['precision_macro']:.4f} "
                        f"| recall_macro={valid_metrics['recall_macro']:.4f} | qwk={valid_metrics['qwk']:.4f} "
                        f"| mae={valid_metrics['mae']:.4f}"
                    )
                    if valid_metrics["ovr_roc_auc_macro"] is not None:
                        log(
                            f"        ovr_roc_auc_macro={valid_metrics['ovr_roc_auc_macro']:.4f} "
                            f"| ovr_pr_auc_macro={valid_metrics['ovr_pr_auc_macro']:.4f}"
                        )

                    improved = early_stopping(valid_metrics["macro_f1"], model)
                    if improved:
                        best_val_macro = valid_metrics["macro_f1"]
                        best_epoch = epoch + 1
                        log(f"  -> best macro_f1 updated to {best_val_macro:.4f} (epoch {best_epoch})")

                    if early_stopping.early_stop:
                        log(f"  -> early stopping triggered at epoch {epoch + 1}")
                        break

                if best_path.exists():
                    model.load_state_dict(torch.load(best_path, map_location=device))

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
                    f"| top3={test_top['top3'] * 100:.2f}% | acc={test_metrics['acc'] * 100:.2f}% "
                    f"| bal_acc={test_metrics['balanced_acc'] * 100:.2f}% | macro_f1={test_metrics['macro_f1']:.4f} "
                    f"| weighted_f1={test_metrics['weighted_f1']:.4f} | precision_macro={test_metrics['precision_macro']:.4f} "
                    f"| recall_macro={test_metrics['recall_macro']:.4f} | qwk={test_metrics['qwk']:.4f} "
                    f"| mae={test_metrics['mae']:.4f}"
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
                    "loss_name": loss_name,
                    "status": "success",
                    "trained_epochs": trained_epochs,
                    "best_epoch": best_epoch,
                    "best_valid_macro_f1": best_val_macro,
                    "test_loss": test_loss,
                    "test_top1": test_top["top1"],
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
                    "loss_name": loss_name,
                    "status": "failed",
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
