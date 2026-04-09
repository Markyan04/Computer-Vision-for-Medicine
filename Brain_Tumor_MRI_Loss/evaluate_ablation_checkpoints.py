#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate saved Brain Tumor MRI checkpoints for the main ablation settings."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from MECS_old import MECS_VersionA
from brain_tumor_mri_loss_experiment_common import (
    DEFAULT_TOPK,
    SEED,
    ResNet50Baseline,
    ResNet50WithInsertedModule,
    build_mri_dataloaders,
    create_experiment_loss,
    evaluate,
    load_checkpoint_states,
    set_seed,
)


def build_baseline(num_classes: int) -> torch.nn.Module:
    return ResNet50Baseline(num_classes=num_classes)


def build_layer2_mecs(num_classes: int) -> torch.nn.Module:
    module = MECS_VersionA(in_channels=512, out_channels=512)
    return ResNet50WithInsertedModule(
        num_classes=num_classes,
        inserted_module=module,
        insert_after='layer2',
    )


def build_layer3_mecs(num_classes: int) -> torch.nn.Module:
    module = MECS_VersionA(in_channels=1024, out_channels=1024)
    return ResNet50WithInsertedModule(
        num_classes=num_classes,
        inserted_module=module,
        insert_after='layer3',
    )


VARIANT_BUILDERS: Dict[str, Callable[[int], torch.nn.Module]] = {
    'baseline': build_baseline,
    'layer2+MECS': build_layer2_mecs,
    'layer3+MECS': build_layer3_mecs,
}

VARIANT_SCRIPT_STEMS: Dict[str, str] = {
    'baseline': 'ResNet_baseline',
    'layer2+MECS': 'ResNet_layer2+MECS',
    'layer3+MECS': 'ResNet_layer3+MECS',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate saved checkpoints for baseline / MECS ablations on Brain Tumor MRI.',
    )
    parser.add_argument(
        '--variants',
        default='baseline,layer2+MECS,layer3+MECS',
        help="Comma-separated variants. Default: 'baseline,layer2+MECS,layer3+MECS'.",
    )
    parser.add_argument(
        '--losses',
        default='ce,dast',
        help="Comma-separated losses. Default: 'ce,dast'.",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=int(os.getenv('BRAIN_MRI_BATCH_SIZE', '16')),
        help='Evaluation batch size. Default follows BRAIN_MRI_BATCH_SIZE or 16.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=int(os.getenv('BRAIN_MRI_NUM_WORKERS', '0')),
        help='DataLoader num_workers. Default follows BRAIN_MRI_NUM_WORKERS or 0.',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=int(os.getenv('BRAIN_MRI_IMAGE_SIZE', '224')),
        help='Image size. Default follows BRAIN_MRI_IMAGE_SIZE or 224.',
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=float(os.getenv('BRAIN_MRI_VAL_RATIO', '0.10')),
        help='Validation split ratio used to rebuild train statistics. Default 0.10.',
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use, e.g. 'cuda' or 'cpu'. Default picks CUDA when available.",
    )
    parser.add_argument(
        '--output',
        default='',
        help='Optional output CSV path. Defaults to Brain_Tumor_MRI_Loss/logs.',
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(',') if item.strip()]


def main() -> None:
    args = parse_args()
    variants = parse_csv_list(args.variants)
    losses = [item.lower() for item in parse_csv_list(args.losses)]

    invalid_variants = [item for item in variants if item not in VARIANT_BUILDERS]
    if invalid_variants:
        raise ValueError(f'Invalid variants: {invalid_variants}. Valid options: {list(VARIANT_BUILDERS)}')

    invalid_losses = [item for item in losses if item not in {'ce', 'dast'}]
    if invalid_losses:
        raise ValueError(f'Invalid losses: {invalid_losses}. Supported here: ce, dast')

    device = torch.device(args.device)
    data_root = Path(os.getenv('BRAIN_MRI_DATA_ROOT', str(PROJECT_ROOT / 'Brain_Tumor_MRI')))
    train_dir = Path(os.getenv('BRAIN_MRI_TRAIN_DIR', str(data_root / 'Training')))
    test_dir = Path(os.getenv('BRAIN_MRI_TEST_DIR', str(data_root / 'Testing')))
    ckpt_dir = THIS_DIR / 'checkpoints'

    set_seed(SEED)
    data_bundle = build_mri_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=SEED,
    )
    class_counts = np.bincount(
        data_bundle.train_targets,
        minlength=data_bundle.num_classes,
    ).tolist()

    print('=' * 100)
    print(f'Device: {device}')
    print(f'Train dir: {train_dir}')
    print(f'Test dir : {test_dir}')
    print(
        f'Config | batch_size={args.batch_size}, num_workers={args.num_workers}, '
        f'image_size={args.image_size}, val_ratio={args.val_ratio}'
    )
    print(f'Variants: {variants}')
    print(f'Losses  : {losses}')
    print(
        f"Split sizes | train={data_bundle.split_sizes['train']}, "
        f"valid={data_bundle.split_sizes['valid']}, test={data_bundle.split_sizes['test']}"
    )
    print(f'Classes ({data_bundle.num_classes}): {data_bundle.class_names}')
    print(f'Train class counts: {class_counts}')
    print('=' * 100)

    rows = []
    for variant in variants:
        script_stem = VARIANT_SCRIPT_STEMS[variant]
        builder = VARIANT_BUILDERS[variant]

        for loss_name in losses:
            checkpoint_path = ckpt_dir / f'best_{script_stem}_{loss_name}.pt'
            if not checkpoint_path.exists():
                raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

            print('')
            print(f'[EVAL] variant={variant} | loss={loss_name}')
            print(f'       checkpoint={checkpoint_path}')

            set_seed(SEED)
            model = builder(data_bundle.num_classes).to(device)
            feat_dim = model.fc.in_features
            criterion = create_experiment_loss(
                loss_name=loss_name,
                num_classes=data_bundle.num_classes,
                class_counts=class_counts,
                feat_dim=feat_dim,
                device=device,
            )
            load_checkpoint_states(checkpoint_path, model, device, criterion)

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

            row = {
                'variant': variant,
                'loss_name': loss_name,
                'checkpoint_path': str(checkpoint_path),
                'test_loss': test_loss,
                'test_top1': test_top['top1'],
                'test_top2': test_top['top2'],
                'test_acc': test_metrics['acc'],
                'test_balanced_acc': test_metrics['balanced_acc'],
                'test_macro_f1': test_metrics['macro_f1'],
                'test_weighted_f1': test_metrics['weighted_f1'],
                'test_precision_macro': test_metrics['precision_macro'],
                'test_recall_macro': test_metrics['recall_macro'],
                'test_mae': test_metrics['mae'],
                'test_qwk': test_metrics['qwk'],
                'test_ovr_roc_auc_macro': test_metrics['ovr_roc_auc_macro'],
                'test_ovr_pr_auc_macro': test_metrics['ovr_pr_auc_macro'],
            }
            rows.append(row)

            print(
                f"       top1={row['test_top1'] * 100:.2f}% | "
                f"macro_f1={row['test_macro_f1']:.4f} | "
                f"mae={row['test_mae']:.4f} | "
                f"qwk={row['test_qwk']:.4f}"
            )

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(['variant', 'loss_name']).reset_index(drop=True)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_path = THIS_DIR / 'logs' / f"checkpoint_eval_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print('')
    print('Result summary:')
    print(result_df[['variant', 'loss_name', 'test_macro_f1', 'test_mae', 'test_qwk']].to_string(index=False))
    print('')
    print(f'Saved CSV: {output_path}')


if __name__ == '__main__':
    main()
