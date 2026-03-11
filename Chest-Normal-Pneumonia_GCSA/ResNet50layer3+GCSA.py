#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chest X-ray Pneumonia using ResNet50 + GCSA (after layer3)

Directory structure:
dataset_root/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

Features:
- ResNet50 pretrained on ImageNet
- Insert GCSA after layer3
- Use official train / val / test split directly
- Weighted CrossEntropyLoss for class imbalance
- Top-1 / Top-2 accuracy
- Accuracy / Balanced Accuracy
- Macro / Weighted F1
- Macro Precision / Macro Recall
- Confusion Matrix / Classification Report
- Binary ROC-AUC / PR-AUC
- Early Stopping based on validation macro_f1
"""

import os
import time
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
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
# GCSA Module
# =======================
class GCSA(nn.Module):
    def __init__(self, in_channels, rate=4, groups=4):
        super(GCSA, self).__init__()

        hidden_channels = max(1, int(in_channels / rate))
        self.groups = groups

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels)
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()

        if num_channels % groups != 0:
            return x

        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        b, c, h, w = x.shape

        # Channel attention
        x_permute = x.permute(0, 2, 3, 1).reshape(b, -1, c)  # [B, H*W, C]
        x_att_permute = self.channel_attention(x_permute)  # [B, H*W, C]
        x_att_permute = x_att_permute.reshape(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()

        x = x * x_channel_att

        # Channel shuffle
        x = self.channel_shuffle(x, groups=self.groups)

        # Spatial attention
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out


# =======================
# ResNet50 + GCSA(after layer3)
# =======================
class ResNet50_GCSA_Layer3(nn.Module):
    def __init__(self, num_classes, weights=models.ResNet50_Weights.DEFAULT):
        super().__init__()

        backbone = models.resnet50(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # output: 256
        self.layer2 = backbone.layer2  # output: 512
        self.layer3 = backbone.layer3  # output: 1024

        # GCSA inserted after layer3
        self.gcsa = GCSA(in_channels=1024, rate=4, groups=4)

        self.layer4 = backbone.layer4  # output: 2048
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.gcsa(x)  # <- inserted here

        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =======================
# Early Stopping
# =======================
class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, loss_delta=1e-4, save_path="best_resnet50_gcsa_layer3.pt"):
        self.patience = patience
        self.delta = delta
        self.loss_delta = loss_delta

        self.best_score = None
        self.best_loss = None

        self.num_bad_epochs = 0
        self.early_stop = False
        self.save_path = save_path

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def __call__(self, score, val_loss, model):
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
            self.save_checkpoint(score, val_loss, model, is_initial=True)

        elif score > self.best_score + self.delta:
            self.best_score = score
            self.best_loss = val_loss
            self.num_bad_epochs = 0
            self.save_checkpoint(score, val_loss, model, msg="Validation macro_f1 improved.")

        elif abs(score - self.best_score) <= self.delta and val_loss < self.best_loss - self.loss_delta:
            self.best_score = score
            self.best_loss = val_loss
            self.num_bad_epochs = 0
            self.save_checkpoint(score, val_loss, model, msg="Validation loss improved at same macro_f1.")

        else:
            self.num_bad_epochs += 1
            print(f"⚠️ No improvement in metrics. Bad epochs: {self.num_bad_epochs}/{self.patience}")

        if self.num_bad_epochs >= self.patience:
            self.early_stop = True
            print("⏹️ Early stopping triggered.")

    def save_checkpoint(self, score, val_loss, model, is_initial=False, msg=""):
        torch.save(model.state_dict(), self.save_path)
        if is_initial:
            print(f"✅ Initial best model saved to {self.save_path} (macro_f1={score:.4f}, val_loss={val_loss:.4f})")
        else:
            print(f"✅ {msg} Saved model to {self.save_path} (macro_f1={score:.4f}, val_loss={val_loss:.4f})")


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
DATA_ROOT = r"..\Pneumonia"  # 改成你的数据集根目录
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40

LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4
PATIENCE = 8
EARLY_STOP_DELTA = 1e-4

TOPK = (1, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# Utils
# =======================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    s = end_time - start_time
    return int(s // 60), int(s % 60)


def get_class_weights_from_dataset(dataset, num_classes):
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets, minlength=num_classes)

    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts.astype(np.float32))

    print("Training class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())

    return torch.tensor(class_weights, dtype=torch.float32)


# =======================
# Metrics
# =======================
def calculate_topk_accuracy(logits, y, ks=(1, 2)):
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

    metrics["roc_auc"] = None
    metrics["pr_auc"] = None

    try:
        if num_classes == 2:
            pos_prob = y_prob[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_true, pos_prob)
            metrics["pr_auc"] = average_precision_score(y_true, pos_prob)
        else:
            y_true_oh = np.eye(num_classes)[y_true]
            metrics["roc_auc"] = roc_auc_score(
                y_true_oh, y_prob, average="macro", multi_class="ovr"
            )
            metrics["pr_auc"] = average_precision_score(
                y_true_oh, y_prob, average="macro"
            )
    except Exception:
        pass

    return metrics


# =======================
# Train / Eval
# =======================
def train_one_epoch(model, loader, optimizer, criterion, scheduler, device, topk=(1, 2)):
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
def evaluate(model, loader, criterion, device, num_classes, class_names, topk=(1, 2)):
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
    print("Starting Chest X-ray Pneumonia ResNet50 + GCSA(layer3) ...")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not os.path.exists(d):
            print(f"Directory not found: {d}")
            return

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
    valid_dataset = ImageFolder(VAL_DIR, transform=eval_transform)
    test_dataset = ImageFolder(TEST_DIR, transform=eval_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Classes: {class_names}")
    print(f"Class-to-index: {train_dataset.class_to_idx}")
    print(f"Split sizes | train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")

    class_weights = get_class_weights_from_dataset(train_dataset, num_classes).to(device)

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

    print("Loading pretrained ResNet50 + GCSA(after layer3)...")
    model = ResNet50_GCSA_Layer3(num_classes=num_classes).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if name.startswith("fc"):
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

    best_path = "best_resnet50_gcsa_layer3_pneumonia.pt"
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
            + f" | weighted_f1={valid_metrics['weighted_f1']:.4f}"
            + f" | precision_macro={valid_metrics['precision_macro']:.4f}"
            + f" | recall_macro={valid_metrics['recall_macro']:.4f}"
        )

        if valid_metrics["roc_auc"] is not None:
            print(
                f"         roc_auc={valid_metrics['roc_auc']:.4f}"
                f" | pr_auc={valid_metrics['pr_auc']:.4f}"
            )

        early_stopping(valid_metrics["macro_f1"], valid_loss, model)

        if early_stopping.early_stop:
            print(f"🛑 Training stopped early at epoch {epoch + 1}.")
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

    if test_metrics["roc_auc"] is not None:
        print(
            f"     roc_auc={test_metrics['roc_auc']:.4f}"
            f" | pr_auc={test_metrics['pr_auc']:.4f}"
        )

    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(test_metrics["classification_report"])

    print("\nChest X-ray Pneumonia ResNet50 + GCSA(layer3) completed!")


if __name__ == "__main__":
    main()
