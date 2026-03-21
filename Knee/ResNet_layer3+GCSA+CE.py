#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Knee Osteoarthritis Severity Grading Baseline using ResNet(layer3) + GCSA + CE
ImageFolder split version (Standard Cross-Entropy Loss, Hardcoded Class Names)

Dataset structure example:
Knee_Osteoarthritis/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
├── val/
...
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
import torchvision.datasets as datasets

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
# 导入 GCSA 模块
# =======================================================
from GCSA import GCSA

warnings.filterwarnings("ignore")


# =======================
# Early Stopping
# =======================
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, save_path="best_resnet50_gcsa_layer3_knee_oa.pt"):
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_score": score,
                },
                self.save_path,
            )
            print(f" Initial best model saved to {self.save_path} (QWK={score:.4f})")

        elif score > self.best_score + self.delta:
            self.best_score = score
            self.num_bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_score": score,
                },
                self.save_path,
            )
            print(f" Validation improved. Saved best model to {self.save_path} (QWK={score:.4f})")

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
DATA_ROOT = r"../Knee_Osteoarthritis"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4
PATIENCE = 15
EARLY_STOP_DELTA = 1e-4

TOPK = (1, 2, 3)

# ===== 临床含义 =====
CLASS_NAMES = ["0_Normal", "1_Doubtful", "2_Mild", "3_Moderate", "4_Severe"]
NUM_CLASSES = len(CLASS_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# Utils
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


def compute_eval_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    metrics["mae"] = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
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


def print_dataset_stats(name, dataset):
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)

    print(f"\n{name} samples: {len(dataset)}")
    print(f"{name} class distribution:")
    for i, c in enumerate(class_counts):
        print(f"  {CLASS_NAMES[i]}: {c}")


# =======================
# Data
# =======================
def make_dataloaders():
    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "val")
    test_dir = os.path.join(DATA_ROOT, "test")
    auto_test_dir = os.path.join(DATA_ROOT, "auto_test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"train dir not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"val dir not found: {val_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"test dir not found: {test_dir}")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    assert len(
        train_dataset.classes) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, but found {len(train_dataset.classes)} folders!"

    auto_test_dataset = None
    if os.path.exists(auto_test_dir):
        auto_test_dataset = datasets.ImageFolder(auto_test_dir, transform=test_transform)

    print("Class mapping from folders:", train_dataset.class_to_idx)
    print_dataset_stats("Train", train_dataset)
    print_dataset_stats("Valid", valid_dataset)
    print_dataset_stats("Test", test_dataset)
    if auto_test_dataset is not None:
        print_dataset_stats("AutoTest", auto_test_dataset)

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

    auto_test_loader = None
    if auto_test_dataset is not None:
        auto_test_loader = data.DataLoader(
            auto_test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    return train_loader, valid_loader, test_loader, auto_test_loader, train_dataset


# =======================
# Custom Model: ResNet50 + GCSA
# =======================
class CustomResNet50GCSA(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomResNet50GCSA, self).__init__()

        # Load pre-trained ResNet50
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Stem
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # Stages
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3

        # [MODIFIED]: GCSA Module (Output of layer3 has 1024 channels)
        self.gcsa = GCSA(in_channels=1024)

        self.layer4 = base_model.layer4

        # Head
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # [MODIFIED]: Injecting GCSA after layer 3
        x = self.gcsa(x)

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
def evaluate(model, loader, criterion, device, topk=(1, 2, 3)):
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


# =======================
# Main
# =======================
def main():
    print("Starting ResNet(layer3)+GCSA+CE Knee Osteoarthritis Baseline...")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(DATA_ROOT):
        print(f"DATA_ROOT not found: {DATA_ROOT}")
        return

    train_loader, valid_loader, test_loader, auto_test_loader, train_dataset = make_dataloaders()

    print(
        f"\nSplit sizes | train={len(train_loader.dataset)}, "
        f"valid={len(valid_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
        + (f", auto_test={len(auto_test_loader.dataset)}" if auto_test_loader is not None else "")
    )

    print("Loading ResNet50 with GCSA module at layer3...")
    model = CustomResNet50GCSA(num_classes=NUM_CLASSES)
    model = model.to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    class_counts = np.bincount(train_dataset.targets, minlength=NUM_CLASSES)
    print(f"\n[INFO] Class counts in training set: {class_counts}")
    print("[INFO] Using Standard CrossEntropyLoss (No class weights).")

    criterion = nn.CrossEntropyLoss()

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        # [MODIFIED]: 将新添加的 gcsa 层和全连接层 fc 一起放入 head_params 赋予更大的学习率
        if "fc" in name or "gcsa" in name:
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

    best_path = "best_resnet50_gcsa_layer3_knee_oa.pt"
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
            model, valid_loader, criterion, device, topk=TOPK
        )

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"\nEpoch {epoch + 1:02d}/{EPOCHS} | Time {epoch_mins}m {epoch_secs}s")
        print(
            f"  Train | loss={train_loss:.4f} | "
            + " | ".join([f'{k}={v * 100:.2f}%' for k, v in train_top.items()])
        )
        print(
            f"  Valid | loss={valid_loss:.4f} | "
            + " | ".join([f'{k}={v * 100:.2f}%' for k, v in valid_top.items()])
            + f" | acc={valid_metrics['acc'] * 100:.2f}%"
            + f" | bal_acc={valid_metrics['balanced_acc'] * 100:.2f}%"
            + f" | macro_f1={valid_metrics['macro_f1']:.4f}"
            + f" | qwk={valid_metrics['qwk']:.4f}"
            + f" | mae={valid_metrics['mae']:.4f}"
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
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Best validation QWK: {ckpt['best_score']:.4f}")

    test_loss, test_top, test_metrics = evaluate(
        model, test_loader, criterion, device, topk=TOPK
    )

    print(
        f"\nTest | loss={test_loss:.4f} | "
        + " | ".join([f'{k}={v * 100:.2f}%' for k, v in test_top.items()])
        + f" | acc={test_metrics['acc'] * 100:.2f}%"
        + f" | bal_acc={test_metrics['balanced_acc'] * 100:.2f}%"
        + f" | macro_f1={test_metrics['macro_f1']:.4f}"
        + f" | qwk={test_metrics['qwk']:.4f}"
        + f" | mae={test_metrics['mae']:.4f}"
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

    if auto_test_loader is not None:
        print("\nEvaluating on auto_test set...")
        auto_test_loss, auto_test_top, auto_test_metrics = evaluate(
            model, auto_test_loader, criterion, device, topk=TOPK
        )

        print(
            f"\nAutoTest | loss={auto_test_loss:.4f} | "
            + " | ".join([f'{k}={v * 100:.2f}%' for k, v in auto_test_top.items()])
            + f" | acc={auto_test_metrics['acc'] * 100:.2f}%"
            + f" | bal_acc={auto_test_metrics['balanced_acc'] * 100:.2f}%"
            + f" | macro_f1={auto_test_metrics['macro_f1']:.4f}"
            + f" | qwk={auto_test_metrics['qwk']:.4f}"
            + f" | mae={auto_test_metrics['mae']:.4f}"
            + f" | weighted_f1={auto_test_metrics['weighted_f1']:.4f}"
            + f" | precision_macro={auto_test_metrics['precision_macro']:.4f}"
            + f" | recall_macro={auto_test_metrics['recall_macro']:.4f}"
        )

        if auto_test_metrics["ovr_roc_auc_macro"] is not None:
            print(
                f"         ovr_roc_auc_macro={auto_test_metrics['ovr_roc_auc_macro']:.4f}"
                f" | ovr_pr_auc_macro={auto_test_metrics['ovr_pr_auc_macro']:.4f}"
            )

    print("\nResNet(layer3)+GCSA+CE Knee Osteoarthritis baseline completed!")


if __name__ == "__main__":
    main()
