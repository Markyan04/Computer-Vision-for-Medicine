#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chest X-ray ResNet50 + MDFA (inserted after layer2)

Directory structure:
dataset_root/
├── train/
│   ├── COVID19/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── COVID19/
    ├── NORMAL/
    └── PNEUMONIA/

Features:
- ResNet50 pretrained on ImageNet
- MDFA inserted after layer2
- Train/Val/Test split (val split from train)
- Top-1 / Top-2 / Top-3 accuracy
- Accuracy / Balanced Accuracy
- Macro / Weighted F1
- Macro Precision / Macro Recall
- Confusion Matrix / Classification Report
- Optional ROC-AUC / PR-AUC (OvR macro)
- Early Stopping based on validation macro_f1
"""

import os
import time
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.models as models

from torchvision.datasets import ImageFolder
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
)

warnings.filterwarnings("ignore")


# =======================
# Early Stopping
# =======================
class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, save_path="best_resnet50_mdfa_layer2_chest_xray.pt"):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.num_bad_epochs = 0
        self.early_stop = False
        self.save_path = save_path

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.save_path)
            print(f" Initial best model saved to {self.save_path} (macro_f1={score:.4f})")
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.num_bad_epochs = 0
            torch.save(model.state_dict(), self.save_path)
            print(f" Validation improved. Saved best model to {self.save_path} (macro_f1={score:.4f})")
        else:
            self.num_bad_epochs += 1
            print(f" No improvement in macro_f1. Bad epochs: {self.num_bad_epochs}/{self.patience}")

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
DATA_ROOT = r"..\CPN"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
VAL_RATIO = 0.1

LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4
PATIENCE = 10
EARLY_STOP_DELTA = 1e-4

TOPK = (1, 2, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# Dataset Wrapper for Transform
# =======================
class TransformSubset(data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


# =======================
# MDFA Module
# =======================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        mid_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return w


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        m = self.conv(avg_map)
        m = self.sigmoid(m)
        return m


class MDFA(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, reduction=2):
        super(MDFA, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                dim_in, dim_out, kernel_size=3, stride=1,
                padding=6 * rate, dilation=6 * rate, bias=False
            ),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                dim_in, dim_out, kernel_size=3, stride=1,
                padding=12 * rate, dilation=12 * rate, bias=False
            ),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(
                dim_in, dim_out, kernel_size=3, stride=1,
                padding=18 * rate, dilation=18 * rate, bias=False
            ),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch5_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        cat_channels = dim_out * 5

        self.channel_att = ChannelAttention(cat_channels, reduction=reduction)
        self.spatial_att = SpatialAttention()

        self.conv_out = nn.Sequential(
            nn.Conv2d(cat_channels, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)

        global_feat = F.adaptive_avg_pool2d(x, output_size=1)
        global_feat = self.branch5_conv(global_feat)
        global_feat = self.branch5_bn(global_feat)
        global_feat = self.branch5_relu(global_feat)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=True)

        feature_cat = torch.cat([feat1, feat2, feat3, feat4, global_feat], dim=1)

        ca = self.channel_att(feature_cat)
        sa = self.spatial_att(feature_cat)

        channel_refined = feature_cat * ca
        spatial_refined = feature_cat * sa

        fused = channel_refined + spatial_refined + feature_cat
        out = self.conv_out(fused)
        return out


# =======================
# ResNet50 + MDFA(after layer2)
# =======================
class ResNet50_MDFA_Layer2(nn.Module):
    def __init__(self, num_classes, weights=models.ResNet50_Weights.DEFAULT):
        super().__init__()

        backbone = models.resnet50(weights=weights)

        # stem
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # res blocks
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 在 layer2 后插入 MDFA
        self.mdfa = MDFA(dim_in=512, dim_out=512)

        # head
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)     # [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # [B, 64, 56, 56]

        x = self.layer1(x)    # [B, 256, 56, 56]
        x = self.layer2(x)    # [B, 512, 28, 28]

        x = self.mdfa(x)      # [B, 512, 28, 28]

        x = self.layer3(x)    # [B, 1024, 14, 14]
        x = self.layer4(x)    # [B, 2048, 7, 7]

        x = self.avgpool(x)   # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =======================
# Metrics
# =======================
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


def epoch_time(start_time, end_time):
    s = end_time - start_time
    return int(s // 60), int(s % 60)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =======================
# Main
# =======================
def main():
    print("Starting Chest X-ray ResNet50 + MDFA(layer2) ...")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(TRAIN_DIR):
        print(f"Train directory not found: {TRAIN_DIR}")
        return
    if not os.path.exists(TEST_DIR):
        print(f"Test directory not found: {TEST_DIR}")
        return

    base_train_dataset = ImageFolder(TRAIN_DIR)
    class_names = base_train_dataset.classes
    num_classes = len(class_names)

    print(f"Classes: {class_names}")
    print(f"Class-to-index: {base_train_dataset.class_to_idx}")
    print(f"Train samples total: {len(base_train_dataset)}")

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_size = int(len(base_train_dataset) * VAL_RATIO)
    train_size = len(base_train_dataset) - val_size

    train_subset, val_subset = data.random_split(
        base_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_dataset = TransformSubset(train_subset, transform=train_transform)
    valid_dataset = TransformSubset(val_subset, transform=test_transform)

    base_test_dataset = ImageFolder(TEST_DIR)
    test_dataset = TransformSubset(base_test_dataset, transform=test_transform)

    print(f"Split sizes | train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")

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

    # Model
    print("Loading pretrained ResNet50 + MDFA...")
    model = ResNet50_MDFA_Layer2(num_classes=num_classes).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()

    # 分层学习率
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

    total_steps = EPOCHS * len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_path = "best_resnet50_mdfa_layer2_chest_xray.pt"
    early_stopping = EarlyStopping(
        patience=PATIENCE,
        delta=EARLY_STOP_DELTA,
        save_path=best_path,
    )

    print("Starting training...")
    best_val_f1 = -1.0

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
            + f" | weighted_f1={valid_metrics['weighted_f1']:.4f}"
            + f" | precision_macro={valid_metrics['precision_macro']:.4f}"
            + f" | recall_macro={valid_metrics['recall_macro']:.4f}"
        )

        if valid_metrics["ovr_roc_auc_macro"] is not None:
            print(
                f"         ovr_roc_auc_macro={valid_metrics['ovr_roc_auc_macro']:.4f}"
                f" | ovr_pr_auc_macro={valid_metrics['ovr_pr_auc_macro']:.4f}"
            )

        if valid_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = valid_metrics["macro_f1"]

        early_stopping(valid_metrics["macro_f1"], model)

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

    print("\nChest X-ray ResNet50 + MDFA(layer2) completed!")


if __name__ == "__main__":
    main()