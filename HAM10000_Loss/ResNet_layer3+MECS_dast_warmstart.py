#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Warm-start continue training for HAM10000 layer3+MECS with DAST."""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim.lr_scheduler as lr_scheduler

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from MECS_old import MECS_VersionA
from ham10000_loss_experiment_common import (  # noqa: E402
    DEFAULT_TOPK,
    DualLogger,
    EarlyStopping,
    ResNet50WithInsertedModule,
    build_ham10000_dataloaders,
    build_optimizer_with_groups,
    count_parameters,
    create_medical_loss,
    epoch_time,
    evaluate,
    has_trainable_parameters,
    load_checkpoint_states,
    set_seed,
    train_one_epoch,
)

SEED = 1234
LOSS_NAME = 'dast'
SCRIPT_STEM = 'ResNet_layer3+MECS_dast_warmstart'
DEFAULT_SOURCE_CKPT = THIS_DIR / 'checkpoints' / 'best_ResNet_layer3+MECS_dast.pt'
DEFAULT_TARGET_CKPT = THIS_DIR / 'checkpoints' / 'best_ResNet_layer3+MECS_dast_warmstart.pt'


def build_model(num_classes: int) -> torch.nn.Module:
    module = MECS_VersionA(in_channels=1024, out_channels=1024)
    return ResNet50WithInsertedModule(
        num_classes=num_classes,
        inserted_module=module,
        insert_after='layer3',
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Warm-start continue training from the existing HAM10000 layer3+MECS+dast checkpoint.',
    )
    parser.add_argument(
        '--checkpoint',
        default=str(DEFAULT_SOURCE_CKPT),
        help='Source checkpoint path for warm start.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=int(os.getenv('HAM10000_WARMSTART_EPOCHS', '20')),
        help='Additional epochs to train after warm start. Default: 20.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=int(os.getenv('HAM10000_BATCH_SIZE', '16')),
        help='Batch size. Default follows HAM10000_BATCH_SIZE or 16.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=int(os.getenv('HAM10000_NUM_WORKERS', '2')),
        help='DataLoader workers. Default follows HAM10000_NUM_WORKERS or 2.',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=int(os.getenv('HAM10000_IMAGE_SIZE', '224')),
        help='Image size. Default follows HAM10000_IMAGE_SIZE or 224.',
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=float(os.getenv('HAM10000_BASE_LR', '1e-4')),
        help='Base learning rate. Default follows HAM10000_BASE_LR or 1e-4.',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=int(os.getenv('HAM10000_PATIENCE', '15')),
        help='Early stopping patience for the warm-start run.',
    )
    parser.add_argument(
        '--early-delta',
        type=float,
        default=float(os.getenv('HAM10000_EARLY_DELTA', '1e-4')),
        help='Early stopping delta for the warm-start run.',
    )
    parser.add_argument(
        '--data-dir',
        default=os.getenv('HAM10000_DATA_DIR', str(PROJECT_ROOT / 'ISIC')),
        help='HAM10000 data directory.',
    )
    parser.add_argument(
        '--save-checkpoint',
        default=str(DEFAULT_TARGET_CKPT),
        help='Output checkpoint path for the warm-start best model.',
    )
    parser.add_argument(
        '--tag',
        default='',
        help='Optional extra tag appended to log/summary filenames.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Warm-start checkpoint not found: {checkpoint_path}')

    save_checkpoint_path = Path(args.save_checkpoint)
    if not save_checkpoint_path.is_absolute():
        save_checkpoint_path = (PROJECT_ROOT / save_checkpoint_path).resolve()
    save_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    topk = DEFAULT_TOPK

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag_suffix = f"_{args.tag}" if args.tag else ''
    logs_dir = THIS_DIR / 'logs'
    log_path = logs_dir / f'{SCRIPT_STEM}{tag_suffix}_{timestamp}.log'
    summary_path = logs_dir / f'{SCRIPT_STEM}{tag_suffix}_{timestamp}_summary.csv'
    logger = DualLogger(log_path)
    log = logger.log

    try:
        log('=' * 90)
        log(f'Script: {SCRIPT_STEM}{tag_suffix}')
        log('Mode: warm start continue training')
        log('Module: MECS | Insert after: layer3')
        log(f'Device: {device}')
        if torch.cuda.is_available():
            log(f'CUDA: {torch.cuda.get_device_name(0)}')
        log(f'Data dir: {data_dir}')
        log(f'Warm-start checkpoint: {checkpoint_path}')
        log(f'Output best checkpoint: {save_checkpoint_path}')
        log(
            f'Config | batch_size={args.batch_size}, epochs={args.epochs}, num_workers={args.num_workers}, '
            f'image_size={args.image_size}, base_lr={args.base_lr}, patience={args.patience}'
        )
        log(f'Loss: {LOSS_NAME}')
        log('=' * 90)

        data_bundle = build_ham10000_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            seed=SEED,
            data_dir=data_dir,
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

        model = build_model(data_bundle.num_classes).to(device)
        feat_dim = model.fc.in_features
        log(f'Model params: {count_parameters(model):,} | feat_dim={feat_dim}')

        criterion = create_medical_loss(
            loss_name=LOSS_NAME,
            num_classes=data_bundle.num_classes,
            class_counts=class_counts,
            feat_dim=feat_dim,
            device=device,
        )
        log(f'Criterion: {criterion.__class__.__name__}')
        log(f'Criterion trainable params: {count_parameters(criterion):,}')

        load_checkpoint_states(checkpoint_path, model, device, criterion)
        log('Warm-start weights loaded successfully.')

        extra_modules = None
        if has_trainable_parameters(criterion):
            extra_modules = [('criterion', criterion, 1.0)]

        optimizer, max_lrs = build_optimizer_with_groups(
            model=model,
            base_lr=args.base_lr,
            group_divisors=[
                ('conv1', 10),
                ('bn1', 10),
                ('layer1', 8),
                ('layer2', 6),
                ('layer3', 4),
                ('inserted_module', 3),
                ('layer4', 2),
                ('fc', 1),
            ],
            extra_modules=extra_modules,
        )
        total_steps = args.epochs * len(data_bundle.train_loader)
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
        )

        early_stopping = EarlyStopping(
            patience=args.patience,
            delta=args.early_delta,
            save_path=save_checkpoint_path,
        )

        best_val_macro = -1.0
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
                loss_name=LOSS_NAME,
                topk=topk,
            )
            valid_loss, valid_top, valid_metrics = evaluate(
                model=model,
                loader=data_bundle.valid_loader,
                criterion=criterion,
                device=device,
                loss_name=LOSS_NAME,
                num_classes=data_bundle.num_classes,
                class_names=data_bundle.class_names,
                topk=topk,
            )
            end_t = time.time()
            mins, secs = epoch_time(start_t, end_t)
            trained_epochs = epoch + 1

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
                best_epoch = epoch + 1
                log(f'  -> warm-start best macro_f1 updated to {best_val_macro:.4f} (epoch {best_epoch})')

            if early_stopping.early_stop:
                log(f'  -> early stopping triggered at epoch {epoch + 1}')
                break

        if save_checkpoint_path.exists():
            load_checkpoint_states(save_checkpoint_path, model, device, criterion)
            log('Loaded warm-start best checkpoint for final test evaluation.')

        test_loss, test_top, test_metrics = evaluate(
            model=model,
            loader=data_bundle.test_loader,
            criterion=criterion,
            device=device,
            loss_name=LOSS_NAME,
            num_classes=data_bundle.num_classes,
            class_names=data_bundle.class_names,
            topk=topk,
        )

        log('')
        log(f'[TEST] loss={LOSS_NAME}')
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
        log('  Confusion Matrix:')
        log(str(test_metrics['confusion_matrix']))
        log('  Classification Report:')
        log(str(test_metrics['classification_report']))

        summary_df = pd.DataFrame([{
            'mode': 'warm_start',
            'source_checkpoint': str(checkpoint_path),
            'output_checkpoint': str(save_checkpoint_path),
            'loss_name': LOSS_NAME,
            'status': 'success',
            'trained_epochs': trained_epochs,
            'best_epoch': best_epoch,
            'best_valid_macro_f1': best_val_macro,
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
        }])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

        log('')
        log('=' * 90)
        log(f'Summary CSV saved: {summary_path}')
        log(f'Log file saved: {log_path}')
        log('=' * 90)
    except Exception as exc:
        log(f'[ERROR] warm-start training failed: {exc}')
        log(traceback.format_exc())
        raise
    finally:
        logger.close()


if __name__ == '__main__':
    main()
