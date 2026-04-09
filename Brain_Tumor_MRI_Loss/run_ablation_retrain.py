#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Retrain the main Brain Tumor MRI ablation settings with a single command."""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


THIS_DIR = Path(__file__).resolve().parent
BATCH_LOG_DIR = THIS_DIR / 'batch_logs'
CHECKPOINT_DIR = THIS_DIR / 'checkpoints'
DEFAULT_SCRIPTS = [
    'ResNet_baseline.py',
    'ResNet_layer2+MECS.py',
    'ResNet_layer3+MECS.py',
]
LOSS_ENV_NAME = 'BRAIN_MRI_LOSSES'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Retrain baseline/layer2+MECS/layer3+MECS with ce and dast.',
    )
    parser.add_argument(
        '--scripts',
        default=','.join(DEFAULT_SCRIPTS),
        help='Comma-separated script names. Default runs baseline, layer2+MECS, layer3+MECS.',
    )
    parser.add_argument(
        '--losses',
        default='ce,dast',
        help="Comma-separated losses to run. Default: 'ce,dast'.",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Training batch size. Default: 64.',
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


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(',') if item.strip()]


def build_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env['BRAIN_MRI_BATCH_SIZE'] = str(args.batch_size)
    env[LOSS_ENV_NAME] = args.losses

    if args.epochs is not None:
        env['BRAIN_MRI_EPOCHS'] = str(args.epochs)
    if args.num_workers is not None:
        env['BRAIN_MRI_NUM_WORKERS'] = str(args.num_workers)
    if args.image_size is not None:
        env['BRAIN_MRI_IMAGE_SIZE'] = str(args.image_size)
    if args.base_lr is not None:
        env['BRAIN_MRI_BASE_LR'] = str(args.base_lr)
    if args.data_root:
        env['BRAIN_MRI_DATA_ROOT'] = args.data_root

    return env


def backup_existing_checkpoints(scripts: List[str], losses: List[str], backup_root: Path) -> List[Tuple[Path, Path]]:
    copied: List[Tuple[Path, Path]] = []
    backup_root.mkdir(parents=True, exist_ok=True)

    for script in scripts:
        stem = Path(script).stem
        for loss_name in losses:
            ckpt_path = CHECKPOINT_DIR / f'best_{stem}_{loss_name}.pt'
            if not ckpt_path.exists():
                continue

            target_path = backup_root / ckpt_path.name
            shutil.copy2(ckpt_path, target_path)
            copied.append((ckpt_path, target_path))

    return copied


def stream_run(script_name: str, env: Dict[str, str], run_log_dir: Path) -> int:
    script_path = THIS_DIR / script_name
    if not script_path.exists():
        print(f'[SKIP] script not found: {script_path}')
        return 127

    run_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_log_dir / f'{script_path.stem}.log'

    header = (
        f"\n{'=' * 100}\n"
        f'RUN SCRIPT: {script_name}\n'
        f"LOSSES: {env.get(LOSS_ENV_NAME, '')}\n"
        f"BATCH SIZE: {env.get('BRAIN_MRI_BATCH_SIZE', '')}\n"
        f"START TIME: {datetime.now().isoformat(timespec='seconds')}\n"
        f'LOG FILE: {log_path}\n'
        f"{'=' * 100}\n"
    )
    print(header, end='')

    with open(log_path, 'w', encoding='utf-8') as fp:
        fp.write(header)
        fp.flush()

        proc = subprocess.Popen(
            [sys.executable, script_name],
            cwd=str(THIS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')
            fp.write(line)
            fp.flush()

        proc.wait()
        code = int(proc.returncode)

        footer = (
            f'\nEXIT CODE: {code}\n'
            f"END TIME: {datetime.now().isoformat(timespec='seconds')}\n"
            f"{'=' * 100}\n"
        )
        print(footer, end='')
        fp.write(footer)
        fp.flush()

    return code


def main() -> None:
    args = parse_args()
    scripts = parse_csv_list(args.scripts)
    losses = [item.lower() for item in parse_csv_list(args.losses)]
    env = build_env(args)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_log_dir = BATCH_LOG_DIR / f'ablation_retrain_{timestamp}'

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
        backup_dir = CHECKPOINT_DIR / 'backups' / f'ablation_retrain_{timestamp}'
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
