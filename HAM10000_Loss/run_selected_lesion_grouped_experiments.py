#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train only four HAM10000 experiments with lesion-level grouped splits:

1. ResNet_baseline + CE
2. ResNet_baseline + DAST
3. ResNet_layer3+MECS + CE
4. ResNet_layer3+MECS + DAST

This file does not modify the original training pipeline. It creates a
separate runner so the original image-level split code remains untouched.
"""

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from MECS_old import MECS_VersionA  # noqa: E402
from ham10000_loss_experiment_common import (  # noqa: E402
    DEFAULT_TOPK,
    SEED,
    DataBundle,
    DualLogger,
    EarlyStopping,
    ISICDataset,
    ResNet50Baseline,
    ResNet50WithInsertedModule,
    TransformSubset,
    _build_valid_dataframe,
    build_optimizer_with_groups,
    count_parameters,
    create_medical_loss,
    epoch_time,
    evaluate,
    get_pretrained_resnet50_state,
    has_trainable_parameters,
    load_checkpoint_states,
    load_pretrained_resnet50_backbone,
    sanitize_run_tag,
    set_seed,
    train_one_epoch,
)


@dataclass
class GroupedSplitBundle:
    data_bundle: DataBundle
    split_lesion_counts: Dict[str, int]
    split_manifest: pd.DataFrame


EXPERIMENTS = [
    {
        'script_stem': 'ResNet_baseline',
        'loss_name': 'ce',
        'module_name': 'Baseline',
        'insert_after': 'none',
        'model_builder': lambda num_classes: ResNet50Baseline(num_classes=num_classes),
        'optimizer_group_divisors': [
            ('conv1', 10),
            ('bn1', 10),
            ('layer1', 8),
            ('layer2', 6),
            ('layer3', 4),
            ('layer4', 2),
            ('fc', 1),
        ],
    },
    {
        'script_stem': 'ResNet_baseline',
        'loss_name': 'dast',
        'module_name': 'Baseline',
        'insert_after': 'none',
        'model_builder': lambda num_classes: ResNet50Baseline(num_classes=num_classes),
        'optimizer_group_divisors': [
            ('conv1', 10),
            ('bn1', 10),
            ('layer1', 8),
            ('layer2', 6),
            ('layer3', 4),
            ('layer4', 2),
            ('fc', 1),
        ],
    },
    {
        'script_stem': 'ResNet_layer3+MECS',
        'loss_name': 'ce',
        'module_name': 'MECS',
        'insert_after': 'layer3',
        'model_builder': lambda num_classes: ResNet50WithInsertedModule(
            num_classes=num_classes,
            inserted_module=MECS_VersionA(in_channels=1024, out_channels=1024),
            insert_after='layer3',
        ),
        'optimizer_group_divisors': [
            ('conv1', 10),
            ('bn1', 10),
            ('layer1', 8),
            ('layer2', 6),
            ('layer3', 4),
            ('inserted_module', 3),
            ('layer4', 2),
            ('fc', 1),
        ],
    },
    {
        'script_stem': 'ResNet_layer3+MECS',
        'loss_name': 'dast',
        'module_name': 'MECS',
        'insert_after': 'layer3',
        'model_builder': lambda num_classes: ResNet50WithInsertedModule(
            num_classes=num_classes,
            inserted_module=MECS_VersionA(in_channels=1024, out_channels=1024),
            insert_after='layer3',
        ),
        'optimizer_group_divisors': [
            ('conv1', 10),
            ('bn1', 10),
            ('layer1', 8),
            ('layer2', 6),
            ('layer3', 4),
            ('inserted_module', 3),
            ('layer4', 2),
            ('fc', 1),
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run four selected HAM10000 experiments with lesion-level grouped splits.',
    )
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--data-dir', default=os.getenv('HAM10000_DATA_DIR', str(PROJECT_ROOT / 'ISIC')))
    parser.add_argument('--batch-size', type=int, default=int(os.getenv('HAM10000_BATCH_SIZE', '16')))
    parser.add_argument('--epochs', type=int, default=int(os.getenv('HAM10000_EPOCHS', '50')))
    parser.add_argument('--num-workers', type=int, default=int(os.getenv('HAM10000_NUM_WORKERS', '2')))
    parser.add_argument('--image-size', type=int, default=int(os.getenv('HAM10000_IMAGE_SIZE', '224')))
    parser.add_argument('--base-lr', type=float, default=float(os.getenv('HAM10000_BASE_LR', '1e-4')))
    parser.add_argument('--patience', type=int, default=int(os.getenv('HAM10000_PATIENCE', '10')))
    parser.add_argument('--early-delta', type=float, default=float(os.getenv('HAM10000_EARLY_DELTA', '1e-4')))
    parser.add_argument('--seed', type=int, default=int(os.getenv('HAM10000_GROUPED_SEED', str(SEED))))
    parser.add_argument('--test-ratio', type=float, default=float(os.getenv('HAM10000_GROUPED_TEST_RATIO', '0.20')))
    parser.add_argument(
        '--valid-ratio',
        type=float,
        default=float(os.getenv('HAM10000_GROUPED_VALID_RATIO', '0.10')),
        help='Validation ratio applied within the lesion-level training pool.',
    )
    parser.add_argument('--dast-tau', type=float, default=float(os.getenv('HAM10000_DAST_TAU', '1.0')))
    parser.add_argument('--dast-gamma', type=float, default=float(os.getenv('HAM10000_DAST_GAMMA', '1.5')))
    parser.add_argument('--run-tag', default=os.getenv('HAM10000_RUN_TAG', ''))
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == 'cpu':
        return torch.device('cpu')
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_grouped_ham10000_dataloaders(
    batch_size: int,
    num_workers: int,
    image_size: int,
    seed: int,
    data_dir: Path,
    test_ratio: float,
    valid_ratio: float,
) -> GroupedSplitBundle:
    valid_df = _build_valid_dataframe(data_dir)
    if 'lesion_id' not in valid_df.columns:
        raise KeyError('HAM10000 metadata must contain lesion_id for grouped splitting.')

    lesion_label_counts = valid_df.groupby('lesion_id')['dx'].nunique()
    inconsistent_lesions = lesion_label_counts[lesion_label_counts > 1]
    if not inconsistent_lesions.empty:
        examples = inconsistent_lesions.index.tolist()[:5]
        raise RuntimeError(
            f'Found lesion_id entries with multiple labels, examples: {examples}'
        )

    class_names = sorted(valid_df['dx'].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    num_classes = len(class_names)

    lesion_df = (
        valid_df.groupby('lesion_id', as_index=False)
        .agg(dx=('dx', 'first'), image_count=('image_id', 'size'))
        .sort_values('lesion_id')
        .reset_index(drop=True)
    )

    lesion_ids = lesion_df['lesion_id'].to_numpy()
    lesion_targets = lesion_df['dx'].map(label_to_idx).to_numpy()
    train_lesions, test_lesions = train_test_split(
        lesion_ids,
        test_size=test_ratio,
        stratify=lesion_targets,
        random_state=seed,
    )

    lesion_df_indexed = lesion_df.set_index('lesion_id')
    train_lesion_targets = lesion_df_indexed.loc[train_lesions, 'dx'].map(label_to_idx).to_numpy()
    train_lesions, valid_lesions = train_test_split(
        train_lesions,
        test_size=valid_ratio,
        stratify=train_lesion_targets,
        random_state=seed,
    )

    split_lesions = {
        'train': set(train_lesions.tolist()),
        'valid': set(valid_lesions.tolist()),
        'test': set(test_lesions.tolist()),
    }

    split_col = pd.Series(index=valid_df.index, dtype='object')
    for split_name, lesion_ids_in_split in split_lesions.items():
        mask = valid_df['lesion_id'].isin(lesion_ids_in_split)
        split_col.loc[mask] = split_name

    if split_col.isna().any():
        raise RuntimeError('Grouped split assignment failed for some HAM10000 samples.')

    split_manifest = valid_df.copy()
    split_manifest['split'] = split_col.values

    train_idx = np.flatnonzero(split_manifest['split'].eq('train').to_numpy())
    valid_idx = np.flatnonzero(split_manifest['split'].eq('valid').to_numpy())
    test_idx = np.flatnonzero(split_manifest['split'].eq('test').to_numpy())

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
    train_dataset = TransformSubset(base_dataset, train_idx, transform=train_tf)
    valid_dataset = TransformSubset(base_dataset, valid_idx, transform=eval_tf)
    test_dataset = TransformSubset(base_dataset, test_idx, transform=eval_tf)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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

    train_targets = np.asarray([base_dataset.targets[idx] for idx in train_idx], dtype=np.int64)
    data_bundle = DataBundle(
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
    split_lesion_counts = {
        'train': len(split_lesions['train']),
        'valid': len(split_lesions['valid']),
        'test': len(split_lesions['test']),
    }
    return GroupedSplitBundle(
        data_bundle=data_bundle,
        split_lesion_counts=split_lesion_counts,
        split_manifest=split_manifest,
    )


def save_summary(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding='utf-8-sig')


def format_ratio(count: int, total: int) -> str:
    if total <= 0:
        return '0.00%'
    return f'{(count / total) * 100:.2f}%'


def main() -> None:
    args = parse_args()
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError('--test-ratio must be between 0 and 1.')
    if not (0.0 < args.valid_ratio < 1.0):
        raise ValueError('--valid-ratio must be between 0 and 1.')

    run_tag = sanitize_run_tag(args.run_tag)
    run_suffix = f'_{run_tag}' if run_tag else ''
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir).expanduser().resolve()

    os.environ['HAM10000_DAST_TAU'] = str(args.dast_tau)
    os.environ['HAM10000_DAST_GAMMA'] = str(args.dast_gamma)

    logs_dir = THIS_DIR / 'logs' / 'lesion_grouped'
    ckpt_dir = THIS_DIR / 'checkpoints' / 'lesion_grouped'
    log_path = logs_dir / f'selected_lesion_grouped{run_suffix}_{timestamp}.log'
    summary_path = logs_dir / f'selected_lesion_grouped{run_suffix}_{timestamp}_summary.csv'
    split_manifest_path = logs_dir / f'selected_lesion_grouped{run_suffix}_{timestamp}_splits.csv'

    logger = DualLogger(log_path)
    log = logger.log

    try:
        set_seed(args.seed)
        grouped_bundle = build_grouped_ham10000_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            seed=args.seed,
            data_dir=data_dir,
            test_ratio=args.test_ratio,
            valid_ratio=args.valid_ratio,
        )
        data_bundle = grouped_bundle.data_bundle
        grouped_bundle.split_manifest.to_csv(split_manifest_path, index=False, encoding='utf-8-sig')

        total_images = sum(data_bundle.split_sizes.values())
        total_lesions = sum(grouped_bundle.split_lesion_counts.values())

        log('=' * 90)
        log('Script: selected_lesion_grouped_experiments')
        if run_tag:
            log(f'Run tag: {run_tag}')
        log(f'Device: {device}')
        if str(device) == 'cuda':
            log(f'CUDA: {torch.cuda.get_device_name(0)}')
        log(f'Data dir: {data_dir}')
        log(
            f'Config | batch_size={args.batch_size}, epochs={args.epochs}, num_workers={args.num_workers}, '
            f'image_size={args.image_size}, base_lr={args.base_lr}, patience={args.patience}, '
            f'seed={args.seed}, test_ratio={args.test_ratio}, valid_ratio={args.valid_ratio}'
        )
        experiment_names = [f"{item['script_stem']}+{item['loss_name']}" for item in EXPERIMENTS]
        log(f'DAST config | tau={args.dast_tau:.4f}, gamma={args.dast_gamma:.4f}')
        log(f'Experiments: {experiment_names}')
        log('=' * 90)
        log(
            f"Image split sizes | train={data_bundle.split_sizes['train']} ({format_ratio(data_bundle.split_sizes['train'], total_images)}), "
            f"valid={data_bundle.split_sizes['valid']} ({format_ratio(data_bundle.split_sizes['valid'], total_images)}), "
            f"test={data_bundle.split_sizes['test']} ({format_ratio(data_bundle.split_sizes['test'], total_images)})"
        )
        log(
            f"Lesion split sizes | train={grouped_bundle.split_lesion_counts['train']} ({format_ratio(grouped_bundle.split_lesion_counts['train'], total_lesions)}), "
            f"valid={grouped_bundle.split_lesion_counts['valid']} ({format_ratio(grouped_bundle.split_lesion_counts['valid'], total_lesions)}), "
            f"test={grouped_bundle.split_lesion_counts['test']} ({format_ratio(grouped_bundle.split_lesion_counts['test'], total_lesions)})"
        )
        log(f'Classes ({data_bundle.num_classes}): {data_bundle.class_names}')

        class_counts = np.bincount(data_bundle.train_targets, minlength=data_bundle.num_classes).tolist()
        log(f'Train class counts: {class_counts}')
        log(f'Split manifest CSV: {split_manifest_path}')
        log('')
        log('Loading torchvision ResNet50 pretrained weights once...')
        pretrained_state = get_pretrained_resnet50_state(data_bundle.num_classes)
        log(f'Pretrained state keys: {len(pretrained_state)}')

        summary_rows: List[Dict[str, object]] = []

        for experiment in EXPERIMENTS:
            script_stem = experiment['script_stem']
            loss_name = experiment['loss_name']
            model_builder: Callable[[int], torch.nn.Module] = experiment['model_builder']
            optimizer_group_divisors: Sequence[Tuple[str, float]] = experiment['optimizer_group_divisors']
            module_name = experiment['module_name']
            insert_after = experiment['insert_after']
            experiment_name = f'{script_stem}+{loss_name}'

            log('')
            log('#' * 90)
            log(f'Starting experiment: {experiment_name}')
            log(f'Module: {module_name} | Insert after: {insert_after}')
            log('#' * 90)

            try:
                set_seed(args.seed)
                model = model_builder(data_bundle.num_classes)
                loaded_n, total_n = load_pretrained_resnet50_backbone(model, pretrained_state)
                model = model.to(device)
                feat_dim = model.fc.in_features
                log(
                    f'Model params: {count_parameters(model):,} | '
                    f'pretrained loaded: {loaded_n}/{total_n} | feat_dim={feat_dim}'
                )

                criterion = create_medical_loss(
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
                    base_lr=args.base_lr,
                    group_divisors=optimizer_group_divisors,
                    extra_modules=extra_modules,
                )
                total_steps = args.epochs * len(data_bundle.train_loader)
                scheduler = lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lrs,
                    total_steps=total_steps,
                )

                best_path = ckpt_dir / f'best_{script_stem}_{loss_name}{run_suffix}_lesion_grouped.pt'
                early_stopping = EarlyStopping(
                    patience=args.patience,
                    delta=args.early_delta,
                    save_path=best_path,
                )

                best_val_macro = -1.0
                best_val_loss = float('inf')
                best_epoch = 0
                trained_epochs = 0

                for epoch in range(args.epochs):
                    start_t = time.time()
                    train_loss, train_top = train_one_epoch(
                        model=model,
                        loader=data_bundle.train_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        scheduler=scheduler,
                        device=device,
                        loss_name=loss_name,
                        topk=DEFAULT_TOPK,
                    )
                    valid_loss, valid_top, valid_metrics = evaluate(
                        model=model,
                        loader=data_bundle.valid_loader,
                        criterion=criterion,
                        device=device,
                        loss_name=loss_name,
                        num_classes=data_bundle.num_classes,
                        class_names=data_bundle.class_names,
                        topk=DEFAULT_TOPK,
                    )
                    trained_epochs = epoch + 1
                    mins, secs = epoch_time(start_t, time.time())

                    log(f'Epoch {epoch + 1:02d}/{args.epochs} | Time {mins}m{secs}s')
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
                    if valid_metrics['ovr_roc_auc_macro'] is not None:
                        log(
                            f"        ovr_roc_auc_macro={valid_metrics['ovr_roc_auc_macro']:.4f} "
                            f"| ovr_pr_auc_macro={valid_metrics['ovr_pr_auc_macro']:.4f}"
                        )

                    improved = early_stopping(valid_metrics['macro_f1'], model, criterion)
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
                    topk=DEFAULT_TOPK,
                )

                log('')
                log(f'[TEST] {experiment_name}')
                log(
                    f"  Test | loss={test_loss:.4f} | top1={test_top['top1'] * 100:.2f}% "
                    f"| top3={test_top['top3'] * 100:.2f}% | acc={test_metrics['acc'] * 100:.2f}% "
                    f"| bal_acc={test_metrics['balanced_acc'] * 100:.2f}% | macro_f1={test_metrics['macro_f1']:.4f} "
                    f"| weighted_f1={test_metrics['weighted_f1']:.4f} | precision_macro={test_metrics['precision_macro']:.4f} "
                    f"| recall_macro={test_metrics['recall_macro']:.4f} | qwk={test_metrics['qwk']:.4f} "
                    f"| mae={test_metrics['mae']:.4f}"
                )
                if test_metrics['ovr_roc_auc_macro'] is not None:
                    log(
                        f"        ovr_roc_auc_macro={test_metrics['ovr_roc_auc_macro']:.4f} "
                        f"| ovr_pr_auc_macro={test_metrics['ovr_pr_auc_macro']:.4f}"
                    )

                summary_rows.append({
                    'run_tag': run_tag,
                    'experiment_name': experiment_name,
                    'script_stem': script_stem,
                    'module_name': module_name,
                    'insert_after': insert_after,
                    'loss_name': loss_name,
                    'split_mode': 'lesion_grouped',
                    'split_seed': args.seed,
                    'test_ratio': args.test_ratio,
                    'valid_ratio_within_train': args.valid_ratio,
                    'dast_tau': args.dast_tau if loss_name == 'dast' else None,
                    'dast_gamma': args.dast_gamma if loss_name == 'dast' else None,
                    'status': 'success',
                    'trained_epochs': trained_epochs,
                    'best_epoch': best_epoch,
                    'best_valid_macro_f1': best_val_macro,
                    'best_valid_loss': best_val_loss,
                    'test_loss': test_loss,
                    'test_top1': test_top['top1'],
                    'test_top3': test_top['top3'],
                    'test_acc': test_metrics['acc'],
                    'test_balanced_acc': test_metrics['balanced_acc'],
                    'test_macro_f1': test_metrics['macro_f1'],
                    'test_weighted_f1': test_metrics['weighted_f1'],
                    'test_precision_macro': test_metrics['precision_macro'],
                    'test_recall_macro': test_metrics['recall_macro'],
                    'test_qwk': test_metrics['qwk'],
                    'test_mae': test_metrics['mae'],
                    'test_ovr_roc_auc_macro': test_metrics['ovr_roc_auc_macro'],
                    'test_ovr_pr_auc_macro': test_metrics['ovr_pr_auc_macro'],
                    'checkpoint_path': str(best_path),
                })
            except Exception as experiment_exc:
                log(f'[ERROR] experiment={experiment_name} failed: {experiment_exc}')
                log(traceback.format_exc())
                summary_rows.append({
                    'run_tag': run_tag,
                    'experiment_name': experiment_name,
                    'script_stem': script_stem,
                    'module_name': module_name,
                    'insert_after': insert_after,
                    'loss_name': loss_name,
                    'split_mode': 'lesion_grouped',
                    'split_seed': args.seed,
                    'test_ratio': args.test_ratio,
                    'valid_ratio_within_train': args.valid_ratio,
                    'dast_tau': args.dast_tau if loss_name == 'dast' else None,
                    'dast_gamma': args.dast_gamma if loss_name == 'dast' else None,
                    'status': 'failed',
                    'error': str(experiment_exc),
                })

        save_summary(summary_rows, summary_path)
        summary_df = pd.DataFrame(summary_rows)
        log('')
        log('=' * 90)
        log(f'Summary CSV saved: {summary_path}')
        if not summary_df.empty and 'status' in summary_df.columns:
            success_df = summary_df[summary_df['status'] == 'success'].copy()
            if not success_df.empty:
                success_df = success_df.sort_values('test_macro_f1', ascending=False)
                log('Top results by test_macro_f1:')
                for _, row in success_df.iterrows():
                    log(
                        f"  {row['experiment_name']}: macro_f1={row['test_macro_f1']:.4f}, "
                        f"acc={row['test_acc'] * 100:.2f}%, bal_acc={row['test_balanced_acc'] * 100:.2f}%"
                    )
            else:
                log('No successful experiment found.')
        log(f'Log file saved: {log_path}')
        log('=' * 90)
    finally:
        logger.close()


if __name__ == '__main__':
    main()
