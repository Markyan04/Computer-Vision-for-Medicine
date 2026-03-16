#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ISIC (HAM10000) Baseline using SE-ResNet50
- 已经缝合 SEAttention 模块
- Top-1 / Top-3 accuracy (Top-k auto clipped by num_classes)
- Additional metrics: balanced accuracy, macro/weighted F1, confusion matrix, classification report
"""

import os
import time
import random
from collections import namedtuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import init  # 引入 init 用于 SE 模块权重初始化
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
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)


# -----------------------
# 早停机制 (Early Stopping)
# -----------------------
class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, save_path="isic_resnet50_best_model.pt"):
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
            print(f" Initial model saved to {self.save_path} (macro_f1={score:.4f})")
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

        # -----------------------


# Reproducibility
# -----------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# -----------------------
# Dataset
# -----------------------
class ISICDataset(data.Dataset):
    def __init__(self, metadata_df: pd.DataFrame, transform=None):
        self.df = metadata_df.reset_index(drop=True).copy()
        self.transform = transform
        self.labels = sorted(self.df["dx"].unique().tolist())
        self.label_to_idx = {lb: i for i, lb in enumerate(self.labels)}

        self.img_paths = []
        self.targets = []
        for _, row in self.df.iterrows():
            img_path = os.path.join(row["image_dir"], row["image_id"] + ".jpg")
            if os.path.exists(img_path):
                self.img_paths.append(img_path)
                self.targets.append(self.label_to_idx[row["dx"]])

        if len(self.img_paths) == 0:
            raise RuntimeError("No valid images found in ISICDataset. Check image paths and metadata.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        img_path = self.base.img_paths[base_idx]
        y = self.base.targets[base_idx]

        x = Image.open(img_path).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, y


# -----------------------
# 缝合模块 1：SE Net 模块代码
# -----------------------
class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    # -----------------------


# SE-ResNet (custom)
# -----------------------
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

        # 缝合模块 2：在Bottleneck中实例化SE模块，通道数为当前块的输出通道数
        self.se = SEAttention(channel=self.expansion * out_channels, reduction=16)
        self.se.init_weights()  # 初始化SE模块权重

        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # 缝合模块 2：在与残差边相加前，经过SE模块重新校准通道特征
        x = self.se(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.relu(x + identity)
        return x


class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        block, n_blocks, channels = config
        assert len(n_blocks) == len(channels) == 4

        self.in_channels = channels[0]
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, n_blocks[0], channels[0], stride=1)
        self.layer2 = self._make_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self._make_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self._make_layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

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
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.size(0), -1)
        logits = self.fc(h)
        return logits, h


ResNetConfig = namedtuple("ResNetConfig", ["block", "n_blocks", "channels"])
resnet50_config = ResNetConfig(
    block=Bottleneck,
    n_blocks=[3, 4, 6, 3],
    channels=[64, 128, 256, 512],
)


# -----------------------
# Metrics
# -----------------------
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
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
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


# -----------------------
# Train / Eval loops
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion, scheduler, device, topk=(1, 3)):
    model.train()
    epoch_loss = 0.0
    epoch_top = {f"top{k}": 0.0 for k in topk}

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
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
def evaluate(model, loader, criterion, device, num_classes, class_names, topk=(1, 3)):
    model.eval()
    epoch_loss = 0.0
    epoch_top = {f"top{k}": 0.0 for k in topk}

    all_y = []
    all_pred = []
    all_prob = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
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


# -----------------------
# Main
# -----------------------
def main():
    print("Starting ISIC SE-ResNet50 Baseline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    data_dir = "../ISIC"
    metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return

    metadata_df = pd.read_csv(metadata_path)
    print(f"Loaded metadata samples: {len(metadata_df)}")
    print(f"Unique diagnoses in metadata: {sorted(metadata_df['dx'].unique().tolist())}")

    part1_dir = os.path.join(data_dir, "HAM10000_images_part_1")
    part2_dir = os.path.join(data_dir, "HAM10000_images_part_2")
    image_dirs = [d for d in [part1_dir, part2_dir] if os.path.exists(d)]
    if not image_dirs:
        print("No image directories found!")
        return
    print(f"Available image directories: {image_dirs}")

    rows = []
    for _, row in metadata_df.iterrows():
        image_id = row["image_id"]
        found = None
        for d in image_dirs:
            if os.path.exists(os.path.join(d, image_id + ".jpg")):
                found = d
                break
        if found is not None:
            r = row.copy()
            r["image_dir"] = found
            rows.append(r)

    valid_df = pd.DataFrame(rows).reset_index(drop=True)
    if len(valid_df) == 0:
        print("No valid images found after checking disk paths.")
        return

    class_names = sorted(valid_df["dx"].unique().tolist())
    num_classes = len(class_names)
    print(f"Valid samples: {len(valid_df)} | num_classes={num_classes} | classes={class_names}")

    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((pretrained_size, pretrained_size)),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(pretrained_size, padding=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means, std=pretrained_stds),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((pretrained_size, pretrained_size)),
        transforms.CenterCrop(pretrained_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means, std=pretrained_stds),
    ])

    base_dataset = ISICDataset(valid_df, transform=None)
    targets = np.array(base_dataset.targets)

    indices = np.arange(len(base_dataset))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=SEED
    )

    train_targets = targets[train_idx]
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=0.1, stratify=train_targets, random_state=SEED
    )

    train_dataset = TransformSubset(base_dataset, train_idx, transform=train_tf)
    valid_dataset = TransformSubset(base_dataset, valid_idx, transform=test_tf)
    test_dataset = TransformSubset(base_dataset, test_idx, transform=test_tf)

    BATCH_SIZE = 32
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print("Loading pre-trained ResNet50 (torchvision) weights...")
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        tv_model = models.resnet50(weights=weights)
    except Exception:
        tv_model = models.resnet50(weights="IMAGENET1K_V1")

    in_features = tv_model.fc.in_features
    tv_model.fc = nn.Linear(in_features, num_classes)

    model = ResNet(resnet50_config, num_classes)

    # 缝合模块 3：必须将 strict=True 改为 False，因为预训练模型中没有SE模块的权重
    model.load_state_dict(tv_model.state_dict(), strict=False)
    model = model.to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    FOUND_LR = 1e-4
    params = [
        {"params": model.conv1.parameters(), "lr": FOUND_LR / 10},
        {"params": model.bn1.parameters(), "lr": FOUND_LR / 10},
        {"params": model.layer1.parameters(), "lr": FOUND_LR / 8},
        {"params": model.layer2.parameters(), "lr": FOUND_LR / 6},
        {"params": model.layer3.parameters(), "lr": FOUND_LR / 4},
        {"params": model.layer4.parameters(), "lr": FOUND_LR / 2},
        {"params": model.fc.parameters(), "lr": FOUND_LR},
    ]
    optimizer = optim.Adam(params, lr=FOUND_LR)

    criterion = nn.CrossEntropyLoss().to(device)

    EPOCHS = 50
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    max_lrs = [g["lr"] for g in optimizer.param_groups]
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs, total_steps=total_steps)

    TOPK = (1, 3)

    best_valid_macro_f1 = -1.0
    best_path = "isic_seresnet50_model_early_senet.pt"
    early_stopping = EarlyStopping(patience=10, delta=0.0001, save_path=best_path)

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
        m, s = epoch_time(start_time, end_time)

        print(f"\nEpoch {epoch + 1:02}/{EPOCHS} | Time {m}m{s}s")
        print(
            f"  Train | loss={train_loss:.4f} | top1={train_top['top1'] * 100:.2f}% | top3={train_top['top3'] * 100:.2f}%")
        print(
            f"  Valid | loss={valid_loss:.4f} | top1={valid_top['top1'] * 100:.2f}% | top3={valid_top['top3'] * 100:.2f}% "
            f"| bal_acc={valid_metrics['balanced_acc'] * 100:.2f}% | macro_f1={valid_metrics['macro_f1']:.4f}")

        early_stopping(valid_metrics["macro_f1"], model)

        if early_stopping.early_stop:
            print(f" Training stopped early at epoch {epoch + 1} to prevent overfitting.")
            break

    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_top, test_metrics = evaluate(
        model, test_loader, criterion, device, num_classes, class_names, topk=TOPK
    )

    print(f"\nTest | loss={test_loss:.4f} | top1={test_top['top1'] * 100:.2f}% | top3={test_top['top3'] * 100:.2f}% "
          f"| acc={test_metrics['acc'] * 100:.2f}% | bal_acc={test_metrics['balanced_acc'] * 100:.2f}% | macro_f1={test_metrics['macro_f1']:.4f} | weighted_f1={test_metrics['weighted_f1']:.4f}")

    if test_metrics["ovr_roc_auc_macro"] is not None:
        print(
            f"     ovr_roc_auc_macro={test_metrics['ovr_roc_auc_macro']:.4f} | ovr_pr_auc_macro={test_metrics['ovr_pr_auc_macro']:.4f}")

    print("\nConfusion Matrix:\n", test_metrics["confusion_matrix"])
    print("\nClassification Report:\n", test_metrics["classification_report"])

    print("\nISIC SE-ResNet50 Baseline completed!")


if __name__ == "__main__":
    main()
