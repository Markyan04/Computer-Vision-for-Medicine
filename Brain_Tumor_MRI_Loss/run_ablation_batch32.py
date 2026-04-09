#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Retrain Brain Tumor MRI baseline/layer3+MECS ablations with batch_size=32."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from run_ablation_retrain import (
    backup_existing_checkpoints,
    build_env,
    stream_run,
)


THIS_DIR = Path(__file__).resolve().parent
BATCH_LOG_DIR = THIS_DIR / 'batch_logs'
DEFAULT_SCRIPTS = [
    'ResNet_baseline.py',
    'ResNet_layer3+MECS.py',
]
DEFAULT_LOSSES = 'ce,dast'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run Brain Tumor MRI batch_size=32 ablations for baseline and layer3+MECS with ce/dast.',
    )
    parser.add_argument(
        '--scripts',
        default=','.join(DEFAULT_SCRIPTS),
        help='Comma-separated script names. Default runs baseline and layer3+MECS.',
    )
    parser.add_argument(
        '--losses',
        default=DEFAULT_LOSSES,
        help="Comma-separated losses to run. Default: 'ce,dast'.",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size. Default: 32.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Optional override for BRAIN_MRI_EPOCHS.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Optional override for BRAIN_MRI_NUM_WORKERS.',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=None,
        help='Optional override for BRAIN_MRI_IMAGE_SIZE.',
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=None,
        help='Optional override for BRAIN_MRI_BASE_LR.',
    )
    parser.add_argument(
        '--data-root',
        default='',
        help='Optional override for BRAIN_MRI_DATA_ROOT.',
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not back up existing targeted checkpoints before retraining.',
    )
    return parser.parse_args()


def parse_csv_list(raw: str):
    return [item.strip() for item in raw.split(',') if item.strip()]


def main() -> None:
    args = parse_args()
    scripts = parse_csv_list(args.scripts)
    losses = [item.lower() for item in parse_csv_list(args.losses)]
    env = build_env(args)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_log_dir = BATCH_LOG_DIR / f'ablation_batch32_{timestamp}'

    print(f'Batch start: {datetime.now().isoformat(timespec="seconds")}')
    print(f'Working dir: {THIS_DIR}')
    print(f'Scripts: {scripts}')
    print(f'Losses: {losses}')
    print(f"batch_size={env['BRAIN_MRI_BATCH_SIZE']}")
    if 'BRAIN_MRI_EPOCHS' in env:
        print(f"epochs={env['BRAIN_MRI_EPOCHS']}")
    if 'BRAIN_MRI_NUM_WORKERS' in env:
        print(f"num_workers={env['BRAIN_MRI_NUM_WORKERS']}")
    if 'BRAIN_MRI_IMAGE_SIZE' in env:
        print(f"image_size={env['BRAIN_MRI_IMAGE_SIZE']}")
    if 'BRAIN_MRI_BASE_LR' in env:
        print(f"base_lr={env['BRAIN_MRI_BASE_LR']}")
    if 'BRAIN_MRI_DATA_ROOT' in env:
        print(f"data_root={env['BRAIN_MRI_DATA_ROOT']}")
    print(f'Batch logs dir: {run_log_dir}')

    if not args.no_backup:
        backup_dir = THIS_DIR / 'checkpoints' / 'backups' / f'ablation_batch32_{timestamp}'
        copied = backup_existing_checkpoints(scripts, losses, backup_dir)
        if copied:
            print(f'Backed up {len(copied)} checkpoint(s) to: {backup_dir}')
        else:
            print('No existing targeted checkpoints were found to back up.')
    else:
        print('Checkpoint backup disabled by --no-backup.')

    results = []
    for script in scripts:
        code = stream_run(script, env=env, run_log_dir=run_log_dir)
        results.append((script, code))

    print('\nBatch summary')
    print('-' * 100)
    for script, code in results:
        status = 'OK' if code == 0 else 'FAIL'
        print(f'{status:4} | code={code:3d} | {script}')
    print('-' * 100)
    print(f'Batch end: {datetime.now().isoformat(timespec="seconds")}')
    print(f'Batch logs dir: {run_log_dir}')
    print(f'Training logs dir: {THIS_DIR / "logs"}')

    if any(code != 0 for _, code in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
