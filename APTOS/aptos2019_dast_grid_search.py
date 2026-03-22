#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
APTOS 2019 Blindness Detection baseline using ResNet50 and DAST Loss.
Grid-search version for tau / gamma experiments.

Expected dataset structure (mariaherrerot/aptos2019 style):
aptos2019/
├── train_images/
├── val_images/
├── test_images/
├── train.csv
├── valid.csv
└── test.csv

What this script adds:
- Batch experiments over multiple tau / gamma combinations
- Per-run best checkpoint saving
- Validation-based early stopping (QWK)
- Test evaluation for each tau / gamma pair
- Final summary table sorted by test QWK, then MAE
"""

import os
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

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

from medical_losses import DistanceAwareSoftTargetLoss

warnings.filterwarnings("ignore")


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, save_path="best_model.pt"):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.num_bad_epochs = 0
        self.early_stop = False
        self.save_path = save_path

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.save_path)
            print(f" Initial best model saved to {self.save_path} (score={score:.4f})")
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.num_bad_epochs = 0
            torch.save(model.state_dict(), self.save_path)
            print(f" Validation improved. Saved best model to {self.save_path} (score={score:.4f})")
        else:
            self.num_bad_epochs += 1
            print(f" No improvement. Bad epochs: {self.num_bad_epochs}/{self.patience}")

        if self.num_bad_epochs >= self.patience:
            self.early_stop = True
            print("⏹ Early stopping triggered.")


# =======================
# Reproducibility
# =======================
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =======================
# Config
# =======================
DATA_ROOT = r"../APTOS-2019"  # 改成你的数据集根目录
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
VAL_DIR = os.path.join(DATA_ROOT, "val_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VAL_CSV = os.path.join(DATA_ROOT, "valid.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 40

LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4
PATIENCE = 10
EARLY_STOP_DELTA = 1e-4
TOPK = (1, 2, 3)

CLASS_NAMES = ["0", "1", "2", "3", "4"]
NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Grid Search Config =====
# 推荐先试这几组；需要更多组时自行扩展
TAU_VALUES = [0.5, 0.75, 1.0]
GAMMA_VALUES = [1.0, 1.5]
RESULTS_CSV = "aptos2019_dast_grid_results.csv"
BEST_MODEL_DIR = "checkpoints_dast_grid"


class APTOSDataset(data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["diagnosis"])

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_stem_to_path_map(image_dir):
    image_dir = Path(image_dir)
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    stem_to_path = {}

    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_exts:
            if p.stem not in stem_to_path:
                stem_to_path[p.stem] = str(p)

    return stem_to_path


def prepare_split_dataframe(csv_path, image_dir):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    df = pd.read_csv(csv_path)
    expected_cols = {"id_code", "diagnosis"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: {expected_cols}")

    stem_to_path = build_stem_to_path_map(image_dir)
    df["path"] = df["id_code"].astype(str).map(stem_to_path)

    missing = df["path"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} samples in {os.path.basename(csv_path)} could not be matched to images.")
        df = df.dropna(subset=["path"]).copy()

    df["diagnosis"] = df["diagnosis"].astype(int)
    df = df[df["diagnosis"].isin(range(NUM_CLASSES))].copy()

    return df[["id_code", "diagnosis", "path"]].reset_index(drop=True)


def calculate_topk_accuracy(logits, y, ks=(1, 2, 3)):
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


def compute_eval_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    metrics["mae"] = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )

    metrics["ovr_roc_auc_macro"] = None
    metrics["ovr_pr_auc_macro"] = None
    try:
        y_true_oh = np.eye(NUM_CLASSES)[y_true]
        metrics["ovr_roc_auc_macro"] = roc_auc_score(
            y_true_oh, y_prob, average="macro", multi_class="ovr"
        )
        metrics["ovr_pr_auc_macro"] = average_precision_score(
            y_true_oh, y_prob, average="macro"
        )
    except Exception:
        pass

    return metrics


def epoch_time(start_time, end_time):
    s = end_time - start_time
    return int(s // 60), int(s % 60)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_split_stats(name, df):
    print(f"\n{name} samples: {len(df)}")
    print(f"{name} class distribution:")
    print(df["diagnosis"].value_counts().sort_index())


def make_dataloaders():
    train_df = prepare_split_dataframe(TRAIN_CSV, TRAIN_DIR)
    val_df = prepare_split_dataframe(VAL_CSV, VAL_DIR)
    test_df = prepare_split_dataframe(TEST_CSV, TEST_DIR)

    print_split_stats("Train", train_df)
    print_split_stats("Valid", val_df)
    print_split_stats("Test", test_df)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = APTOSDataset(train_df, transform=train_transform)
    valid_dataset = APTOSDataset(val_df, transform=eval_transform)
    test_dataset = APTOSDataset(test_df, transform=eval_transform)

    train_loader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    return train_loader, valid_loader, test_loader, train_df, val_df, test_df


def train_one_epoch(model, loader, optimizer, criterion, scheduler, device, topk=(1, 2, 3)):
    model.train()
    epoch_loss = 0.0
    epoch_top = {f"top{k}": 0.0 for k in topk}

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
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
def evaluate(model, loader, criterion, device, num_classes, class_names, topk=(1, 2, 3)):
    model.eval()
    epoch_loss = 0.0
    epoch_top = {f"top{k}": 0.0 for k in topk}

    all_y = []
    all_pred = []
    all_prob = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
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

    metrics = compute_eval_metrics(y_true, y_pred, y_prob)
    return epoch_loss, epoch_top, metrics


def build_model():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model.to(DEVICE)


def build_optimizer_and_scheduler(model, train_loader_len):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = EPOCHS * train_loader_len
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    return optimizer, scheduler


def format_metrics_line(prefix, loss, top, metrics):
    msg = (
        f"{prefix} | loss={loss:.4f} | "
        + " | ".join([f"{k}={v * 100:.2f}%" for k, v in top.items()])
        + f" | acc={metrics['acc'] * 100:.2f}%"
        + f" | bal_acc={metrics['balanced_acc'] * 100:.2f}%"
        + f" | macro_f1={metrics['macro_f1']:.4f}"
        + f" | qwk={metrics['qwk']:.4f}"
        + f" | mae={metrics['mae']:.4f}"
        + f" | weighted_f1={metrics['weighted_f1']:.4f}"
        + f" | precision_macro={metrics['precision_macro']:.4f}"
        + f" | recall_macro={metrics['recall_macro']:.4f}"
    )
    return msg


def run_single_experiment(tau, gamma, train_loader, valid_loader, test_loader):
    print("\n" + "=" * 88)
    print(f"[RUN] tau={tau} | gamma={gamma}")
    print("=" * 88)

    model = build_model()
    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = DistanceAwareSoftTargetLoss(num_classes=NUM_CLASSES, tau=tau, gamma=gamma).to(DEVICE)
    print("[INFO] Using Distance-Aware Soft Target Loss (DAST)")
    print(f"       - tau={tau}")
    print(f"       - gamma={gamma}")

    optimizer, scheduler = build_optimizer_and_scheduler(model, len(train_loader))

    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    best_path = os.path.join(BEST_MODEL_DIR, f"best_tau_{str(tau).replace('.', 'p')}_gamma_{str(gamma).replace('.', 'p')}.pt")
    early_stopping = EarlyStopping(
        patience=PATIENCE,
        delta=EARLY_STOP_DELTA,
        save_path=best_path,
    )

    best_valid_metrics = None
    best_valid_loss = None
    best_epoch = None

    print("Starting training...")
    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_top = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, DEVICE, topk=TOPK
        )
        valid_loss, valid_top, valid_metrics = evaluate(
            model, valid_loader, criterion, DEVICE, NUM_CLASSES, CLASS_NAMES, topk=TOPK
        )

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"\nEpoch {epoch + 1:02d}/{EPOCHS} | Time {epoch_mins}m {epoch_secs}s")
        print(f"  Train | loss={train_loss:.4f} | " + " | ".join([f"{k}={v * 100:.2f}%" for k, v in train_top.items()]))
        print("  " + format_metrics_line("Valid", valid_loss, valid_top, valid_metrics))

        if valid_metrics["ovr_roc_auc_macro"] is not None:
            print(
                f"         ovr_roc_auc_macro={valid_metrics['ovr_roc_auc_macro']:.4f}"
                f" | ovr_pr_auc_macro={valid_metrics['ovr_pr_auc_macro']:.4f}"
            )

        prev_best = early_stopping.best_score
        early_stopping(valid_metrics["qwk"], model)
        if prev_best is None or valid_metrics["qwk"] > prev_best + EARLY_STOP_DELTA:
            best_valid_metrics = valid_metrics.copy()
            best_valid_loss = valid_loss
            best_epoch = epoch + 1

        if early_stopping.early_stop:
            print(f" Training stopped early at epoch {epoch + 1}.")
            break

    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_loss, test_top, test_metrics = evaluate(
        model, test_loader, criterion, DEVICE, NUM_CLASSES, CLASS_NAMES, topk=TOPK
    )

    print("\n" + format_metrics_line("Test", test_loss, test_top, test_metrics))
    if test_metrics["ovr_roc_auc_macro"] is not None:
        print(
            f"     ovr_roc_auc_macro={test_metrics['ovr_roc_auc_macro']:.4f}"
            f" | ovr_pr_auc_macro={test_metrics['ovr_pr_auc_macro']:.4f}"
        )

    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(test_metrics["classification_report"])

    torch.cuda.empty_cache()

    result = {
        "tau": tau,
        "gamma": gamma,
        "best_epoch": best_epoch,
        "best_model_path": best_path,
        "valid_loss": best_valid_loss,
        "valid_qwk": None if best_valid_metrics is None else best_valid_metrics["qwk"],
        "valid_mae": None if best_valid_metrics is None else best_valid_metrics["mae"],
        "valid_acc": None if best_valid_metrics is None else best_valid_metrics["acc"],
        "valid_balanced_acc": None if best_valid_metrics is None else best_valid_metrics["balanced_acc"],
        "valid_macro_f1": None if best_valid_metrics is None else best_valid_metrics["macro_f1"],
        "test_loss": test_loss,
        "test_top1": test_top.get("top1"),
        "test_top2": test_top.get("top2"),
        "test_top3": test_top.get("top3"),
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
    }
    return result


def print_summary_table(results_df):
    if results_df.empty:
        print("\nNo results to summarize.")
        return

    view_cols = [
        "tau", "gamma", "best_epoch",
        "valid_qwk", "valid_mae",
        "test_acc", "test_balanced_acc", "test_macro_f1",
        "test_qwk", "test_mae", "test_ovr_roc_auc_macro", "test_ovr_pr_auc_macro"
    ]
    print("\n" + "#" * 100)
    print("FINAL SUMMARY (sorted by test_qwk desc, test_mae asc)")
    print("#" * 100)
    print(results_df[view_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    best_row = results_df.iloc[0]
    print("\nBest setting by summary rule:")
    print(
        f"  tau={best_row['tau']}, gamma={best_row['gamma']}"
        f" | test_qwk={best_row['test_qwk']:.4f}"
        f" | test_mae={best_row['test_mae']:.4f}"
        f" | test_acc={best_row['test_acc'] * 100:.2f}%"
        f" | test_macro_f1={best_row['test_macro_f1']:.4f}"
    )


def main():
    print("Starting APTOS2019 ResNet50 DAST Grid Search...")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    for p in [DATA_ROOT, TRAIN_DIR, VAL_DIR, TEST_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not os.path.exists(p):
            print(f"Path not found: {p}")
            return

    train_loader, valid_loader, test_loader, train_df, val_df, test_df = make_dataloaders()
    print(f"\nSplit sizes | train={len(train_df)}, valid={len(val_df)}, test={len(test_df)}")
    print(f"\nGrid search space: tau={TAU_VALUES}, gamma={GAMMA_VALUES}")

    results = []
    for tau in TAU_VALUES:
        for gamma in GAMMA_VALUES:
            result = run_single_experiment(tau, gamma, train_loader, valid_loader, test_loader)
            results.append(result)

            temp_df = pd.DataFrame(results)
            temp_df = temp_df.sort_values(by=["test_qwk", "test_mae"], ascending=[False, True]).reset_index(drop=True)
            temp_df.to_csv(RESULTS_CSV, index=False)
            print(f"\n[INFO] Intermediate results saved to: {RESULTS_CSV}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["test_qwk", "test_mae"], ascending=[False, True]).reset_index(drop=True)
    results_df.to_csv(RESULTS_CSV, index=False)

    print_summary_table(results_df)
    print(f"\nAll grid-search results saved to: {RESULTS_CSV}")
    print("APTOS2019 ResNet50 DAST grid search completed!")


if __name__ == "__main__":
    main()
