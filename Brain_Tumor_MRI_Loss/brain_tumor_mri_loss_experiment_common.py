#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared training and evaluation utilities for Brain_Tumor_MRI loss experiments."""

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
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
CHEST_LOSS_DIR = PROJECT_ROOT / 'chest-x-ray-image_Loss'
for path in (PROJECT_ROOT, CHEST_LOSS_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chest_xray_loss_experiment_common import (  # noqa: E402
    Bottleneck,
    DualLogger,
    ResNet50WithInsertedModule,
    calculate_topk_accuracy,
    count_parameters,
    create_medical_loss,
    epoch_time,
    get_pretrained_resnet50_state,
    has_trainable_parameters,
    load_checkpoint_states,
    load_pretrained_resnet50_backbone,
    set_seed,
)


SEED = 1234
DEFAULT_TOPK = (1, 2)
DEFAULT_LOSS_ORDER = (
    'ce',
    'cb_focal_ce',
    'ordinal_focal_mse',
    'sce',
    'gce',
    'dast',
    'pcol',
    'aom',
)


class EarlyStopping:
    def __init__(self, patience: int, delta: float, loss_delta: float, save_path: Path):
        self.patience = patience
        self.delta = delta
        self.loss_delta = loss_delta
        self.save_path = save_path
        self.best_score: Optional[float] = None
        self.best_loss: Optional[float] = None
        self.num_bad_epochs = 0
        self.early_stop = False
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        score: float,
        val_loss: float,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
    ) -> bool:
        improved = False
        better_score = self.best_score is None or score > self.best_score + self.delta
        tie_score_better_loss = (
            self.best_score is not None
            and abs(score - self.best_score) <= self.delta
            and (self.best_loss is None or val_loss < self.best_loss - self.loss_delta)
        )

        if better_score or tie_score_better_loss:
            self.best_score = score
            self.best_loss = val_loss
            self.num_bad_epochs = 0
            payload = {'model_state': model.state_dict()}
            if criterion is not None and has_trainable_parameters(criterion):
                payload['criterion_state'] = criterion.state_dict()
            torch.save(payload, self.save_path)
            improved = True
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.early_stop = True
        return improved


class SubsetImageDataset(data.Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.targets = [label for _, label in self.samples]

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 3, 64, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 4, 128, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 6, 256, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 3, 512, stride=2)

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
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        h = x.view(x.size(0), -1)
        logits = self.fc(h)
        return logits, h


@dataclass
class DataBundle:
    train_loader: data.DataLoader
    valid_loader: data.DataLoader
    test_loader: data.DataLoader
    train_targets: np.ndarray
    class_names: List[str]
    num_classes: int
    split_sizes: Dict[str, int]


def build_mri_dataloaders(
    train_dir: Path,
    test_dir: Path,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    image_size: int,
    seed: int,
) -> DataBundle:
    if not train_dir.exists():
        raise FileNotFoundError(f'Train directory not found: {train_dir}')
    if not test_dir.exists():
        raise FileNotFoundError(f'Test directory not found: {test_dir}')

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train_dataset = ImageFolder(str(train_dir))
    test_dataset_raw = ImageFolder(str(test_dir))

    class_names = list(full_train_dataset.classes)
    class_to_idx = full_train_dataset.class_to_idx
    num_classes = len(class_names)

    all_samples = full_train_dataset.samples
    all_targets = [label for _, label in all_samples]

    train_samples, valid_samples = train_test_split(
        all_samples,
        test_size=val_ratio,
        random_state=seed,
        stratify=all_targets,
    )

    train_dataset = SubsetImageDataset(train_samples, class_to_idx, transform=train_transform)
    valid_dataset = SubsetImageDataset(valid_samples, class_to_idx, transform=eval_transform)
    test_dataset = SubsetImageDataset(test_dataset_raw.samples, class_to_idx, transform=eval_transform)

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

    train_targets = np.array(train_dataset.targets, dtype=np.int64)

    return DataBundle(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_targets=train_targets,
        class_names=class_names,
        num_classes=num_classes,
        split_sizes={
            'train': len(train_dataset),
            'valid': len(valid_dataset),
            'test': len(test_dataset),
        },
    )


def compute_eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    class_names: Sequence[str],
) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    metrics['classification_report'] = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )

    metrics['ovr_roc_auc_macro'] = None
    metrics['ovr_pr_auc_macro'] = None
    try:
        y_true_oh = np.eye(num_classes)[y_true]
        metrics['ovr_roc_auc_macro'] = roc_auc_score(
            y_true_oh,
            y_prob,
            average='macro',
            multi_class='ovr',
        )
        metrics['ovr_pr_auc_macro'] = average_precision_score(
            y_true_oh,
            y_prob,
            average='macro',
        )
    except Exception:
        pass

    return metrics


def build_optimizer_with_groups(
    model: nn.Module,
    base_lr: float,
    group_divisors: Sequence[Tuple[str, float]],
    extra_modules: Optional[Sequence[Tuple[str, nn.Module, float]]] = None,
) -> Tuple[optim.Optimizer, List[float]]:
    param_groups = []
    max_lrs: List[float] = []

    def add_param_group(module: nn.Module, divisor: float) -> None:
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            return
        lr = base_lr / float(divisor)
        param_groups.append({'params': params, 'lr': lr})
        max_lrs.append(lr)

    for attr_name, divisor in group_divisors:
        add_param_group(getattr(model, attr_name), divisor)

    if extra_modules is not None:
        for _, module, divisor in extra_modules:
            add_param_group(module, divisor)

    optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=1e-4)
    return optimizer, max_lrs


def create_experiment_loss(
    loss_name: str,
    num_classes: int,
    class_counts: List[int],
    feat_dim: int,
    device: torch.device,
) -> nn.Module:
    if loss_name == 'ce':
        return nn.CrossEntropyLoss().to(device)
    return create_medical_loss(
        loss_name=loss_name,
        num_classes=num_classes,
        class_counts=class_counts,
        feat_dim=feat_dim,
        device=device,
    )


def _compute_loss(
    criterion: nn.Module,
    loss_name: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    if loss_name == 'pcol':
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
    topk=(1, 2),
) -> Tuple[float, Dict[str, float]]:
    model.train()
    criterion.train()
    epoch_loss = 0.0
    epoch_top = {f'top{k}': 0.0 for k in topk}

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
    topk=(1, 2),
) -> Tuple[float, Dict[str, float], Dict[str, object]]:
    model.eval()
    criterion.eval()
    epoch_loss = 0.0
    epoch_top = {f'top{k}': 0.0 for k in topk}

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


def resolve_losses_to_run() -> List[str]:
    env_losses = os.getenv('BRAIN_MRI_LOSSES', '').strip()
    if not env_losses:
        return list(DEFAULT_LOSS_ORDER)

    requested = [x.strip().lower() for x in env_losses.split(',') if x.strip()]
    invalid = [x for x in requested if x not in DEFAULT_LOSS_ORDER]
    if invalid:
        raise ValueError(f'Invalid loss names in BRAIN_MRI_LOSSES: {invalid}')
    return requested


def run_brain_tumor_mri_medical_losses_experiments(
    script_stem: str,
    model_builder: Callable[[int], nn.Module],
    optimizer_group_divisors: Sequence[Tuple[str, float]],
    module_name: str,
    insert_after: str,
) -> None:
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = Path(os.getenv('BRAIN_MRI_DATA_ROOT', str(PROJECT_ROOT / 'Brain_Tumor_MRI')))
    train_dir = Path(os.getenv('BRAIN_MRI_TRAIN_DIR', str(data_root / 'Training')))
    test_dir = Path(os.getenv('BRAIN_MRI_TEST_DIR', str(data_root / 'Testing')))

    val_ratio = float(os.getenv('BRAIN_MRI_VAL_RATIO', '0.10'))
    batch_size = int(os.getenv('BRAIN_MRI_BATCH_SIZE', '64'))
    epochs = int(os.getenv('BRAIN_MRI_EPOCHS', '50'))
    num_workers = int(os.getenv('BRAIN_MRI_NUM_WORKERS', '0'))
    image_size = int(os.getenv('BRAIN_MRI_IMAGE_SIZE', '224'))
    base_lr = float(os.getenv('BRAIN_MRI_BASE_LR', '1e-3'))
    patience = int(os.getenv('BRAIN_MRI_PATIENCE', '10'))
    early_delta = float(os.getenv('BRAIN_MRI_EARLY_DELTA', '1e-4'))
    loss_delta = float(os.getenv('BRAIN_MRI_LOSS_DELTA', '1e-4'))
    topk = DEFAULT_TOPK

    losses_to_run = resolve_losses_to_run()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = THIS_DIR / 'logs'
    ckpt_dir = THIS_DIR / 'checkpoints'
    log_path = logs_dir / f'{script_stem}_{timestamp}.log'
    summary_path = logs_dir / f'{script_stem}_{timestamp}_summary.csv'
    logger = DualLogger(log_path)
    log = logger.log

    try:
        log('=' * 90)
        log(f'Script: {script_stem}')
        log(f'Module: {module_name} | Insert after: {insert_after}')
        log(f'Device: {device}')
        if torch.cuda.is_available():
            log(f'CUDA: {torch.cuda.get_device_name(0)}')
        log(f'Train dir: {train_dir}')
        log(f'Test dir : {test_dir}')
        log(
            f'Config | batch_size={batch_size}, epochs={epochs}, num_workers={num_workers}, '
            f'image_size={image_size}, base_lr={base_lr}, val_ratio={val_ratio}, patience={patience}'
        )
        log(f'Losses to run: {losses_to_run}')
        log('=' * 90)

        data_bundle = build_mri_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
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
        log(f'Classes ({data_bundle.num_classes}): {data_bundle.class_names}')

        class_counts = np.bincount(
            data_bundle.train_targets,
            minlength=data_bundle.num_classes,
        ).tolist()
        log(f'Train class counts: {class_counts}')

        log('Loading torchvision ResNet50 pretrained weights once...')
        pretrained_state = get_pretrained_resnet50_state(data_bundle.num_classes)
        log(f'Pretrained state keys: {len(pretrained_state)}')

        summary_rows = []

        for loss_name in losses_to_run:
            log('')
            log('#' * 90)
            log(f'Starting loss: {loss_name}')
            log('#' * 90)

            try:
                set_seed(SEED)
                model = model_builder(data_bundle.num_classes)
                loaded_n, total_n = load_pretrained_resnet50_backbone(model, pretrained_state)
                model = model.to(device)
                feat_dim = model.fc.in_features
                log(
                    f'Model params: {count_parameters(model):,} | '
                    f'pretrained loaded: {loaded_n}/{total_n} | feat_dim={feat_dim}'
                )

                criterion = create_experiment_loss(
                    loss_name=loss_name,
                    num_classes=data_bundle.num_classes,
                    class_counts=class_counts,
                    feat_dim=feat_dim,
                    device=device,
                )
                log(f'Criterion: {criterion.__class__.__name__}')
                criterion_trainable_params = count_parameters(criterion)
                log(f'Criterion trainable params: {criterion_trainable_params:,}')

                extra_modules = None
                if has_trainable_parameters(criterion):
                    extra_modules = [('criterion', criterion, 1.0)]

                optimizer, max_lrs = build_optimizer_with_groups(
                    model=model,
                    base_lr=base_lr,
                    group_divisors=optimizer_group_divisors,
                    extra_modules=extra_modules,
                )
                total_steps = epochs * len(data_bundle.train_loader)
                scheduler = lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lrs,
                    total_steps=total_steps,
                    pct_start=0.1,
                    anneal_strategy='cos',
                )

                best_path = ckpt_dir / f'best_{script_stem}_{loss_name}.pt'
                early_stopping = EarlyStopping(
                    patience=patience,
                    delta=early_delta,
                    loss_delta=loss_delta,
                    save_path=best_path,
                )

                best_val_macro = -1.0
                best_val_loss = float('inf')
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

                    log(f'Epoch {epoch + 1:02d}/{epochs} | Time {mins}m{secs}s')
                    log(
                        f"  Train | loss={train_loss:.4f} | top1={train_top['top1'] * 100:.2f}% "
                        f"| top2={train_top['top2'] * 100:.2f}%"
                    )
                    log(
                        f"  Valid | loss={valid_loss:.4f} | top1={valid_top['top1'] * 100:.2f}% "
                        f"| top2={valid_top['top2'] * 100:.2f}% | acc={valid_metrics['acc'] * 100:.2f}% "
                        f"| bal_acc={valid_metrics['balanced_acc'] * 100:.2f}% "
                        f"| macro_f1={valid_metrics['macro_f1']:.4f} "
                        f"| weighted_f1={valid_metrics['weighted_f1']:.4f} "
                        f"| precision_macro={valid_metrics['precision_macro']:.4f} "
                        f"| recall_macro={valid_metrics['recall_macro']:.4f}"
                    )
                    if valid_metrics['ovr_roc_auc_macro'] is not None:
                        log(
                            f"        ovr_roc_auc_macro={valid_metrics['ovr_roc_auc_macro']:.4f} "
                            f"| ovr_pr_auc_macro={valid_metrics['ovr_pr_auc_macro']:.4f}"
                        )

                    improved = early_stopping(valid_metrics['macro_f1'], valid_loss, model, criterion)
                    if improved:
                        best_val_macro = valid_metrics['macro_f1']
                        best_val_loss = valid_loss
                        best_epoch = epoch + 1
                        log(
                            f'  -> best macro_f1 updated to {best_val_macro:.4f} '
                            f'(epoch {best_epoch}, val_loss={best_val_loss:.4f})'
                        )

                    if early_stopping.early_stop:
                        log(f'  -> early stopping triggered at epoch {epoch + 1}')
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

                log('')
                log(f'[TEST] loss={loss_name}')
                log(
                    f"  Test | loss={test_loss:.4f} | top1={test_top['top1'] * 100:.2f}% "
                    f"| top2={test_top['top2'] * 100:.2f}% | acc={test_metrics['acc'] * 100:.2f}% "
                    f"| bal_acc={test_metrics['balanced_acc'] * 100:.2f}% "
                    f"| macro_f1={test_metrics['macro_f1']:.4f} "
                    f"| weighted_f1={test_metrics['weighted_f1']:.4f} "
                    f"| precision_macro={test_metrics['precision_macro']:.4f} "
                    f"| recall_macro={test_metrics['recall_macro']:.4f}"
                )
                if test_metrics['ovr_roc_auc_macro'] is not None:
                    log(
                        f"        ovr_roc_auc_macro={test_metrics['ovr_roc_auc_macro']:.4f} "
                        f"| ovr_pr_auc_macro={test_metrics['ovr_pr_auc_macro']:.4f}"
                    )
                log('  Confusion Matrix:')
                log(str(test_metrics['confusion_matrix']))
                log('  Classification Report:')
                log(str(test_metrics['classification_report']))

                summary_rows.append({
                    'loss_name': loss_name,
                    'status': 'success',
                    'trained_epochs': trained_epochs,
                    'best_epoch': best_epoch,
                    'best_valid_macro_f1': best_val_macro,
                    'best_valid_loss': best_val_loss,
                    'test_loss': test_loss,
                    'test_top1': test_top['top1'],
                    'test_top2': test_top['top2'],
                    'test_acc': test_metrics['acc'],
                    'test_balanced_acc': test_metrics['balanced_acc'],
                    'test_macro_f1': test_metrics['macro_f1'],
                    'test_weighted_f1': test_metrics['weighted_f1'],
                    'test_precision_macro': test_metrics['precision_macro'],
                    'test_recall_macro': test_metrics['recall_macro'],
                    'test_ovr_roc_auc_macro': test_metrics['ovr_roc_auc_macro'],
                    'test_ovr_pr_auc_macro': test_metrics['ovr_pr_auc_macro'],
                    'checkpoint_path': str(best_path),
                })
            except Exception as loss_exc:
                log(f'[ERROR] loss={loss_name} failed: {loss_exc}')
                log(traceback.format_exc())
                summary_rows.append({
                    'loss_name': loss_name,
                    'status': 'failed',
                    'error': str(loss_exc),
                })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

        log('')
        log('=' * 90)
        log(f'Summary CSV saved: {summary_path}')
        if not summary_df.empty and 'status' in summary_df.columns:
            success_df = summary_df[summary_df['status'] == 'success'].copy()
            if not success_df.empty and 'test_macro_f1' in success_df.columns:
                success_df = success_df.sort_values('test_macro_f1', ascending=False)
                log('Top results by test_macro_f1:')
                for _, row in success_df.iterrows():
                    log(
                        f"  {row['loss_name']}: macro_f1={row['test_macro_f1']:.4f}, "
                        f"acc={row['test_acc']:.4f}, top1={row['test_top1']:.4f}"
                    )
        log('=' * 90)
    finally:
        logger.close()
