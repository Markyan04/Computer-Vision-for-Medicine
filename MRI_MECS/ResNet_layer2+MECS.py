#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brain Tumor MRI Baseline using ResNet50 + MECS (after layer2)

Dataset structure:
dataset_root/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/

Features:
- ResNet50 pretrained on ImageNet
- MECS_VersionA inserted after layer2
- Split validation set from Training directory
- Weighted CrossEntropyLoss for class imbalance
- Top-1 / Top-2 accuracy
- Accuracy / Balanced Accuracy
- Macro / Weighted F1
- Macro Precision / Macro Recall
- Confusion Matrix / Classification Report
- Multi-class ROC-AUC / PR-AUC (OvR macro)
- Early Stopping based on validation macro_f1
"""

import os
import time
import random
import warnings

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

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
)

warnings.filterwarnings("ignore")


# =======================
# Early Stopping
# =======================
class EarlyStopping:
    def __init__(self, patience=8, delta=0.0, loss_delta=1e-4, save_path="best_resnet50_mecs_brain_mri.pt"):
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
            print(f" No improvement in metrics. Bad epochs: {self.num_bad_epochs}/{self.patience}")

        if self.num_bad_epochs >= self.patience:
            self.early_stop = True
            print("⏹ Early stopping triggered.")

    def save_checkpoint(self, score, val_loss, model, is_initial=False, msg=""):
        torch.save(model.state_dict(), self.save_path)
        if is_initial:
            print(f" Initial best model saved to {self.save_path} (macro_f1={score:.4f}, val_loss={val_loss:.4f})")
        else:
            print(f" {msg} Saved model to {self.save_path} (macro_f1={score:.4f}, val_loss={val_loss:.4f})")


# =======================
# MECS Module (Replaces GCSA)
# =======================
def global_median_pooling(x):
    """全局中位数池化"""
    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled


class ChannelAttention_VersionA(nn.Module):
    """【版本 A：先 Sigmoid，再相加。权重范围 [0, 3]】"""

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention_VersionA, self).__init__()
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, inputs):
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        median_pool = global_median_pooling(inputs)

        # 1. Avg 路径 (包含 Sigmoid)
        avg_out = self.fc2(F.relu(self.fc1(avg_pool), inplace=True))
        avg_out = torch.sigmoid(avg_out)

        # 2. Max 路径 (包含 Sigmoid)
        max_out = self.fc2(F.relu(self.fc1(max_pool), inplace=True))
        max_out = torch.sigmoid(max_out)

        # 3. Median 路径 (包含 Sigmoid)
        median_out = self.fc2(F.relu(self.fc1(median_pool), inplace=True))
        median_out = torch.sigmoid(median_out)

        # 4. 最后相加 (完全符合架构图)
        out = avg_out + max_out + median_out
        return out


class MECS_VersionA(nn.Module):
    def __init__(self, in_channels, out_channels, channel_attention_reduce=4):
        super(MECS_VersionA, self).__init__()
        assert in_channels == out_channels, "Input and output channels must be the same"

        self.channel_attention = ChannelAttention_VersionA(
            input_channels=in_channels,
            internal_neurons=max(1, in_channels // channel_attention_reduce)
        )

        self.initial_depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.depth_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels),
        ])

        # 修复了权重共享问题的三个独立卷积层
        self.pre_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.spatial_att_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.post_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        # 1. 预处理
        x = self.pre_conv(inputs)
        x = self.act(x)

        # 2. 通道注意力
        channel_att_vec = self.channel_attention(x)
        x_ca = channel_att_vec * x

        # 3. 空间注意力
        initial_out = self.initial_depth_conv(x_ca)
        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        spatial_out = sum(spatial_outs)

        # 补齐了架构图中的残差连接
        spatial_out = spatial_out + x_ca

        # 补齐了空间 Sigmoid
        spatial_att = torch.sigmoid(self.spatial_att_conv(spatial_out))
        out = spatial_att * x_ca

        # 4. 后处理
        out = self.post_conv(out)
        return out


# =======================
# ResNet50 + MECS
# =======================
class ResNet50_MECS_Layer2(nn.Module):
    def __init__(self, num_classes, weights=models.ResNet50_Weights.DEFAULT, channel_reduce=4):
        super().__init__()

        backbone = models.resnet50(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2

        # ResNet50 layer2 output channels = 512
        self.mecs = MECS_VersionA(
            in_channels=512,
            out_channels=512,
            channel_attention_reduce=channel_reduce
        )

        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.mecs(x)  # 插在 layer2 后

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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
DATA_ROOT = r"..\MRI"  # 改成你的数据集根目录
TRAIN_DIR = os.path.join(DATA_ROOT, "Training")
TEST_DIR = os.path.join(DATA_ROOT, "Testing")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

VAL_RATIO = 0.15

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


def get_class_weights_from_targets(targets, num_classes):
    class_counts = np.bincount(targets, minlength=num_classes)
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts.astype(np.float32))

    print("Training class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())

    return torch.tensor(class_weights, dtype=torch.float32)


# =======================
# Dataset wrapper
# =======================
class SubsetImageDataset(data.Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.targets = [label for _, label in self.samples]

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


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
    print("Starting Brain Tumor MRI ResNet50 + MECS (after layer2)...")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    for d in [TRAIN_DIR, TEST_DIR]:
        if not os.path.exists(d):
            print(f"Directory not found: {d}")
            return

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    full_train_dataset = ImageFolder(TRAIN_DIR)
    test_dataset_raw = ImageFolder(TEST_DIR)

    class_names = full_train_dataset.classes
    class_to_idx = full_train_dataset.class_to_idx
    num_classes = len(class_names)

    print(f"Classes: {class_names}")
    print(f"Class-to-index: {class_to_idx}")

    all_samples = full_train_dataset.samples
    all_targets = [label for _, label in all_samples]

    train_samples, valid_samples = train_test_split(
        all_samples,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=all_targets,
    )

    train_dataset = SubsetImageDataset(train_samples, class_to_idx, transform=train_transform)
    valid_dataset = SubsetImageDataset(valid_samples, class_to_idx, transform=eval_transform)
    test_dataset = SubsetImageDataset(test_dataset_raw.samples, class_to_idx, transform=eval_transform)

    print(f"Split sizes | train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")

    class_weights = get_class_weights_from_targets(train_dataset.targets, num_classes).to(device)

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

    print("Loading pretrained ResNet50 + MECS...")
    model = ResNet50_MECS_Layer2(
        num_classes=num_classes,
        weights=models.ResNet50_Weights.DEFAULT,
        channel_reduce=4,
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if ("fc" in name) or ("mecs" in name):
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

    best_path = "best_resnet50_mecs_layer2_brain_mri.pt"
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

    if test_metrics["roc_auc"] is not None:
        print(
            f"     roc_auc={test_metrics['roc_auc']:.4f}"
            f" | pr_auc={test_metrics['pr_auc']:.4f}"
        )

    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(test_metrics["classification_report"])

    print("\nBrain Tumor MRI ResNet50 + MECS completed!")


if __name__ == "__main__":
    main()
