#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diabetic Retinopathy Detection Baseline using ResNet50 + MECS (Layer3) + Loss4 (DAST)
Patient-level split version

Dataset structure example:
Diabetic_Retinopathy_Detection/
├── colored_images/
│   ├── Mild/
│   ├── Moderate/
│   ├── No_DR/
│   ├── Proliferate_DR/
│   └── Severe/
└── trainLabels.csv

Features:
- ResNet50 pretrained on ImageNet
- MECS module inserted after layer3 (channels=1024)
- Build dataset from CSV + auto-match image files in colored_images
- Patient-level split using GroupShuffleSplit (prevents left/right eye leakage)
- Early stopping based on QWK
- DistanceAwareSoftTargetLoss (Loss4) applied for ordinal disease grading
"""

import os
import time
import copy
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

from sklearn.model_selection import GroupShuffleSplit
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

# =======================================================
# 导入医用损失函数库中的 DistanceAwareSoftTargetLoss
# =======================================================
from medical_losses import DistanceAwareSoftTargetLoss

# =======================================================
# 导入 MECS 模块
# =======================================================
from MECS_old import MECS_VersionA

warnings.filterwarnings("ignore")


# =======================
# Early Stopping
# =======================
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, save_path="best_resnet50_dr_loss4.pt"):
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
DATA_ROOT = r"../Diabetic_Retinopathy_Detection"  # 改成你的路径
IMAGE_ROOT = os.path.join(DATA_ROOT, "colored_images")
CSV_PATH = os.path.join(DATA_ROOT, "trainLabels.csv")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

VAL_RATIO = 0.1
TEST_RATIO = 0.1

LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4
PATIENCE = 10
EARLY_STOP_DELTA = 1e-4

TOPK = (1, 2, 3)

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
IDX_TO_CLASS = {0: "No_DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferate_DR"}
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# Dataset
# =======================
class DRDataset(data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["level"])

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# =======================
# Utils
# =======================
def build_image_stem_mapping(image_root):
    image_root = Path(image_root)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    stem_to_path = {}
    duplicates = []

    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            stem = path.stem
            if stem in stem_to_path:
                duplicates.append(stem)
            else:
                stem_to_path[stem] = str(path)

    if duplicates:
        print(f"Warning: found duplicate image stems, example: {duplicates[:10]}")
        print("Using the first matched path for duplicated stems.")

    return stem_to_path


def extract_patient_id(image_name: str) -> str:
    stem = Path(str(image_name)).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def prepare_dataframe(csv_path, image_root):
    df = pd.read_csv(csv_path)

    if "image" not in df.columns or "level" not in df.columns:
        raise ValueError("CSV must contain columns: image, level")

    stem_to_path = build_image_stem_mapping(image_root)
    df["path"] = df["image"].map(stem_to_path)

    missing = df["path"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} samples in CSV could not be matched to image files.")
        df = df.dropna(subset=["path"]).copy()

    df["level"] = df["level"].astype(int)
    df = df[df["level"].isin([0, 1, 2, 3, 4])].copy()

    df["patient_id"] = df["image"].apply(extract_patient_id)

    def folder_name(p):
        return Path(p).parent.name

    df["folder_class"] = df["path"].apply(folder_name)
    mismatch_df = df[df["folder_class"].map(CLASS_TO_IDX) != df["level"]]
    if len(mismatch_df) > 0:
        print(f"Warning: found {len(mismatch_df)} samples where folder class != csv label.")
        print("Will trust CSV labels for training.")

    return df[["image", "level", "path", "patient_id"]].reset_index(drop=True)


def _group_split(df, test_size, random_state):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["patient_id"].values
    idx_train, idx_test = next(splitter.split(df, groups=groups))
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()


def verify_patient_disjoint(train_df, val_df, test_df):
    train_patients = set(train_df["patient_id"].unique())
    val_patients = set(val_df["patient_id"].unique())
    test_patients = set(test_df["patient_id"].unique())

    assert train_patients.isdisjoint(val_patients), "Train/Val patient leakage detected!"
    assert train_patients.isdisjoint(test_patients), "Train/Test patient leakage detected!"
    assert val_patients.isdisjoint(test_patients), "Val/Test patient leakage detected!"


def calculate_topk_accuracy(logits, y, ks=(1, 3)):
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


def compute_eval_metrics(y_true, y_pred, y_prob, num_classes, class_names):
    metrics = {}
    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
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


def epoch_time(start_time, end_time):
    s = end_time - start_time
    return int(s // 60), int(s % 60)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_split_stats(name, df):
    print(f"\n{name} samples: {len(df)}")
    print(f"{name} unique patients: {df['patient_id'].nunique()}")
    print(f"{name} class distribution:")
    print(df["level"].value_counts().sort_index())


def make_dataloaders():
    df = prepare_dataframe(CSV_PATH, IMAGE_ROOT)

    print(f"Total matched samples: {len(df)}")
    print(f"Total unique patients: {df['patient_id'].nunique()}")
    print("Overall class distribution:")
    print(df["level"].value_counts().sort_index())

    # 先按患者划分 trainval / test
    trainval_df, test_df = _group_split(df, test_size=TEST_RATIO, random_state=SEED)

    # 再按患者划分 train / val
    val_relative_ratio = VAL_RATIO / (1.0 - TEST_RATIO)
    train_df, val_df = _group_split(trainval_df, test_size=val_relative_ratio, random_state=SEED)

    verify_patient_disjoint(train_df, val_df, test_df)

    print_split_stats("Train", train_df)
    print_split_stats("Valid", val_df)
    print_split_stats("Test", test_df)

    print(
        f"\nUnique patients | train={train_df['patient_id'].nunique()}, "
        f"valid={val_df['patient_id'].nunique()}, "
        f"test={test_df['patient_id'].nunique()}"
    )

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = DRDataset(train_df, transform=train_transform)
    valid_dataset = DRDataset(val_df, transform=test_transform)
    test_dataset = DRDataset(test_df, transform=test_transform)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader, train_df, val_df, test_df


# =======================
# Custom Model: ResNet50 + MECS (Layer 3)
# =======================
class ResNet50_MECS_Layer3(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet50_MECS_Layer3, self).__init__()
        # 1. 加载预训练的 ResNet50
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)

        # 2. 提取并分离 ResNet50 的各个层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # 3. 插入 MECS 模块 (ResNet50 layer3 输出通道数为 1024)
        self.mecs = MECS_VersionA(in_channels=1024, out_channels=1024)

        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # 4. 重置全连接层
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 在 layer3 和 layer4 之间通过 MECS 模块
        x = self.mecs(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# =======================
# Train / Eval
# =======================
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

    metrics = compute_eval_metrics(y_true, y_pred, y_prob, num_classes, class_names)
    return epoch_loss, epoch_top, metrics


# =======================
# Main
# =======================
def main():
    print("Starting Diabetic Retinopathy ResNet50 + MECS (Layer 3) + Loss4 Baseline...")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(DATA_ROOT):
        print(f"DATA_ROOT not found: {DATA_ROOT}")
        return
    if not os.path.exists(IMAGE_ROOT):
        print(f"IMAGE_ROOT not found: {IMAGE_ROOT}")
        return
    if not os.path.exists(CSV_PATH):
        print(f"CSV_PATH not found: {CSV_PATH}")
        return

    train_loader, valid_loader, test_loader, train_df, val_df, test_df = make_dataloaders()

    num_classes = len(CLASS_NAMES)
    class_names = CLASS_NAMES

    print(f"\nSplit sizes | train={len(train_df)}, valid={len(val_df)}, test={len(test_df)}")

    print("Loading pretrained ResNet50 and injecting MECS after layer3...")
    model = ResNet50_MECS_Layer3(num_classes=num_classes)
    model = model.to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    # =======================================================
    # 替换为 Distance-Aware Soft Target Loss
    # =======================================================
    print(f"Initializing DistanceAwareSoftTargetLoss for {num_classes} ordinal classes...")
    criterion = DistanceAwareSoftTargetLoss(
        num_classes=num_classes,
        tau=1.0,
        gamma=1.5
    ).to(device)
    # =======================================================

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        # 将分类头 (fc) 和新初始化的 MECS 模块 (mecs) 都放入 head_params 中，使用较大的学习率
        if "fc" in name or "mecs" in name:
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

    total_steps = EPOCHS * len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # 更改保存路径，以便于区分这是带有 MECS 的模型
    best_path = "best_resnet50_dr_mecs_layer3_loss4_patient_split.pt"
    early_stopping = EarlyStopping(
        patience=PATIENCE,
        delta=EARLY_STOP_DELTA,
        save_path=best_path,
    )

    print("Starting training...")
    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_top = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, topk=TOPK
        )

        valid_loss, valid_top, valid_metrics = evaluate(
            model, valid_loader, criterion, device, num_classes, class_names, topk=TOPK
        )

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"\nEpoch {epoch + 1:02d}/{EPOCHS} | Time {epoch_mins}m {epoch_secs}s")
        print(
            f"  Train | loss={train_loss:.4f} | "
            + " | ".join([f"{k}={v * 100:.2f}%" for k, v in train_top.items()])
        )
        print(
            f"  Valid | loss={valid_loss:.4f} | "
            + " | ".join([f"{k}={v * 100:.2f}%" for k, v in valid_top.items()])
            + f" | acc={valid_metrics['acc'] * 100:.2f}%"
            + f" | bal_acc={valid_metrics['balanced_acc'] * 100:.2f}%"
            + f" | macro_f1={valid_metrics['macro_f1']:.4f}"
            + f" | qwk={valid_metrics['qwk']:.4f}"
            + f" | weighted_f1={valid_metrics['weighted_f1']:.4f}"
            + f" | precision_macro={valid_metrics['precision_macro']:.4f}"
            + f" | recall_macro={valid_metrics['recall_macro']:.4f}"
        )

        if valid_metrics["ovr_roc_auc_macro"] is not None:
            print(
                f"         ovr_roc_auc_macro={valid_metrics['ovr_roc_auc_macro']:.4f}"
                f" | ovr_pr_auc_macro={valid_metrics['ovr_pr_auc_macro']:.4f}"
            )

        early_stopping(valid_metrics["qwk"], model)

        if early_stopping.early_stop:
            print(f" Training stopped early at epoch {epoch + 1}.")
            break

    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_top, test_metrics = evaluate(
        model, test_loader, criterion, device, num_classes, class_names, topk=TOPK
    )

    print(
        f"\nTest | loss={test_loss:.4f} | "
        + " | ".join([f"{k}={v * 100:.2f}%" for k, v in test_top.items()])
        + f" | acc={test_metrics['acc'] * 100:.2f}%"
        + f" | bal_acc={test_metrics['balanced_acc'] * 100:.2f}%"
        + f" | macro_f1={test_metrics['macro_f1']:.4f}"
        + f" | qwk={test_metrics['qwk']:.4f}"
        + f" | weighted_f1={test_metrics['weighted_f1']:.4f}"
        + f" | precision_macro={test_metrics['precision_macro']:.4f}"
        + f" | recall_macro={test_metrics['recall_macro']:.4f}"
    )

    if test_metrics["ovr_roc_auc_macro"] is not None:
        print(
            f"     ovr_roc_auc_macro={test_metrics['ovr_roc_auc_macro']:.4f}"
            f" | ovr_pr_auc_macro={test_metrics['ovr_pr_auc_macro']:.4f}"
        )

    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(test_metrics["classification_report"])

    print("\nDiabetic Retinopathy ResNet50 + MECS (Layer 3) + Loss4 baseline completed!")


if __name__ == "__main__":
    main()
