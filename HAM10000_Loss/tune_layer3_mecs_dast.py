#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Grid search DAST tau/gamma for HAM10000 ResNet layer3+MECS."""

import argparse
import csv
import math
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence


THIS_DIR = Path(__file__).resolve().parent
LOG_DIR = THIS_DIR / 'logs'
BATCH_LOG_DIR = THIS_DIR / 'batch_logs'
TRAIN_SCRIPT = 'ResNet_layer3+MECS.py'
LOSS_NAME = 'dast'
LOSS_ENV_NAME = 'HAM10000_LOSSES'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Tune DAST tau/gamma for HAM10000 ResNet layer3+MECS.',
    )
    parser.add_argument(
        '--taus',
        default='0.5,1.0,1.5',
        help="Comma-separated tau values. Default: '0.5,1.0,1.5'.",
    )
    parser.add_argument(
        '--gammas',
        default='1.0,1.5,2.0',
        help="Comma-separated gamma values. Default: '0.0,1.0,1.5,2.0'.",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=int(os.getenv('HAM10000_BATCH_SIZE', '32')),
        help='Training batch size. Default reads HAM10000_BATCH_SIZE or falls back to 32.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Optional override for HAM10000_EPOCHS.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Optional override for HAM10000_NUM_WORKERS.',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=None,
        help='Optional override for HAM10000_IMAGE_SIZE.',
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=None,
        help='Optional override for HAM10000_BASE_LR.',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help='Optional override for HAM10000_PATIENCE.',
    )
    parser.add_argument(
        '--early-delta',
        type=float,
        default=None,
        help='Optional override for HAM10000_EARLY_DELTA.',
    )
    parser.add_argument(
        '--data-dir',
        default='',
        help='Optional override for HAM10000_DATA_DIR.',
    )
    parser.add_argument(
        '--tag-prefix',
        default='dast_tune',
        help="Prefix used in run tags and checkpoint names. Default: 'dast_tune'.",
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the planned tau/gamma combinations without launching training.',
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError('At least one numeric value is required.')
    return values


def float_slug(value: float) -> str:
    text = f'{value:g}'
    return text.replace('-', 'm').replace('.', 'p')


def build_run_tag(tag_prefix: str, tau: float, gamma: float) -> str:
    return f'{tag_prefix}_tau{float_slug(tau)}_gamma{float_slug(gamma)}'


def build_env(args: argparse.Namespace, tau: float, gamma: float, run_tag: str) -> Dict[str, str]:
    env = os.environ.copy()
    env[LOSS_ENV_NAME] = LOSS_NAME
    env['HAM10000_BATCH_SIZE'] = str(args.batch_size)
    env['HAM10000_DAST_TAU'] = str(tau)
    env['HAM10000_DAST_GAMMA'] = str(gamma)
    env['HAM10000_RUN_TAG'] = run_tag

    if args.epochs is not None:
        env['HAM10000_EPOCHS'] = str(args.epochs)
    if args.num_workers is not None:
        env['HAM10000_NUM_WORKERS'] = str(args.num_workers)
    if args.image_size is not None:
        env['HAM10000_IMAGE_SIZE'] = str(args.image_size)
    if args.base_lr is not None:
        env['HAM10000_BASE_LR'] = str(args.base_lr)
    if args.patience is not None:
        env['HAM10000_PATIENCE'] = str(args.patience)
    if args.early_delta is not None:
        env['HAM10000_EARLY_DELTA'] = str(args.early_delta)
    if args.data_dir:
        env['HAM10000_DATA_DIR'] = args.data_dir

    return env


def safe_float(value: object, default: float = float('nan')) -> float:
    if value in (None, ''):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def format_metric(value: object, digits: int = 4) -> str:
    parsed = safe_float(value)
    if math.isnan(parsed):
        return 'n/a'
    return f'{parsed:.{digits}f}'


def load_summary_row(summary_path: Path) -> Optional[Dict[str, object]]:
    with open(summary_path, 'r', encoding='utf-8-sig', newline='') as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    if not rows:
        return None

    for row in rows:
        if str(row.get('loss_name', '')).strip().lower() == LOSS_NAME:
            return dict(row)
    return None


def run_one_trial(
    tau: float,
    gamma: float,
    env: Dict[str, str],
    run_tag: str,
    run_log_dir: Path,
) -> Dict[str, object]:
    run_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_log_dir / f'{run_tag}.log'

    header = (
        f"\n{'=' * 100}\n"
        f'RUN TAG: {run_tag}\n'
        f'SCRIPT: {TRAIN_SCRIPT}\n'
        f'LOSS: {LOSS_NAME}\n'
        f'tau={tau}, gamma={gamma}\n'
        f"BATCH SIZE: {env.get('HAM10000_BATCH_SIZE', '')}\n"
        f"START TIME: {datetime.now().isoformat(timespec='seconds')}\n"
        f'LOG FILE: {log_path}\n'
        f"{'=' * 100}\n"
    )
    print(header, end='')

    summary_path: Optional[Path] = None

    with open(log_path, 'w', encoding='utf-8') as fp:
        fp.write(header)
        fp.flush()

        proc = subprocess.Popen(
            [sys.executable, TRAIN_SCRIPT],
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
            if line.startswith('Summary CSV saved: '):
                summary_path = Path(line.split(': ', 1)[1].strip())

        proc.wait()
        return_code = int(proc.returncode)

        footer = (
            f'\nEXIT CODE: {return_code}\n'
            f"END TIME: {datetime.now().isoformat(timespec='seconds')}\n"
            f"{'=' * 100}\n"
        )
        print(footer, end='')
        fp.write(footer)
        fp.flush()

    result: Dict[str, object] = {
        'run_tag': run_tag,
        'tau': tau,
        'gamma': gamma,
        'return_code': return_code,
        'batch_log_path': str(log_path),
        'summary_path': str(summary_path) if summary_path else '',
        'status': 'failed',
    }

    if return_code != 0:
        result['error'] = f'training exited with code {return_code}'
        return result

    if summary_path is None or not summary_path.exists():
        result['error'] = 'summary csv not found after training'
        return result

    row = load_summary_row(summary_path)
    if row is None:
        result['error'] = 'no dast row found in summary csv'
        return result

    summary_status = str(row.get('status', 'success'))
    row.update({key: value for key, value in result.items() if key != 'status'})
    row['status'] = summary_status
    return row


def sort_success_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    success_rows = [row for row in rows if str(row.get('status', '')).lower() == 'success']
    return sorted(
        success_rows,
        key=lambda row: (
            -safe_float(row.get('best_valid_macro_f1'), default=float('-inf')),
            safe_float(row.get('best_valid_loss'), default=float('inf')),
            -safe_float(row.get('test_macro_f1'), default=float('-inf')),
        ),
    )


def print_ranked_results(rows: Sequence[Dict[str, object]]) -> None:
    ranked_rows = sort_success_rows(rows)
    if not ranked_rows:
        print('No successful trials to rank.')
        return

    print('\nTop successful trials (ranked by best_valid_macro_f1, best_valid_loss):')
    for row in ranked_rows:
        print(
            f"tau={row.get('tau')} | gamma={row.get('gamma')} | "
            f"best_valid_macro_f1={format_metric(row.get('best_valid_macro_f1'))} | "
            f"best_valid_loss={format_metric(row.get('best_valid_loss'))} | "
            f"test_macro_f1={format_metric(row.get('test_macro_f1'))} | "
            f"test_acc={format_metric(row.get('test_acc'))} | "
            f"test_mae={format_metric(row.get('test_mae'))} | "
            f"test_qwk={format_metric(row.get('test_qwk'))} | "
            f"checkpoint={row.get('checkpoint_path', '')}"
        )


def write_results(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    preferred_fields = [
        'run_tag',
        'tau',
        'gamma',
        'status',
        'return_code',
        'best_valid_macro_f1',
        'best_valid_loss',
        'test_macro_f1',
        'test_acc',
        'test_mae',
        'test_qwk',
        'trained_epochs',
        'best_epoch',
        'test_loss',
        'test_top1',
        'test_top3',
        'test_balanced_acc',
        'test_weighted_f1',
        'test_precision_macro',
        'test_recall_macro',
        'test_ovr_roc_auc_macro',
        'test_ovr_pr_auc_macro',
        'dast_tau',
        'dast_gamma',
        'checkpoint_path',
        'summary_path',
        'batch_log_path',
        'error',
    ]

    discovered_fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in discovered_fields:
                discovered_fields.append(key)

    fieldnames = [field for field in preferred_fields if field in discovered_fields]
    fieldnames.extend(field for field in discovered_fields if field not in fieldnames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    taus = parse_float_list(args.taus)
    gammas = parse_float_list(args.gammas)

    if any(tau <= 0 for tau in taus):
        raise ValueError('All tau values must be > 0.')
    if any(gamma < 0 for gamma in gammas):
        raise ValueError('All gamma values must be >= 0.')

    combos = list(product(taus, gammas))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_log_dir = BATCH_LOG_DIR / f'layer3_mecs_dast_tuning_{timestamp}'
    output_path = LOG_DIR / f'ResNet_layer3+MECS_dast_tuning_{timestamp}.csv'

    print(f'Tuning script: {THIS_DIR / TRAIN_SCRIPT}')
    print(f'Trials: {len(combos)}')
    print(f'taus={taus}')
    print(f'gammas={gammas}')
    print(f'batch_size={args.batch_size}')
    if args.epochs is not None:
        print(f'epochs={args.epochs}')
    if args.base_lr is not None:
        print(f'base_lr={args.base_lr}')
    if args.data_dir:
        print(f'data_dir={args.data_dir}')
    print(f'Batch logs dir: {run_log_dir}')
    print(f'Results CSV: {output_path}')

    if args.dry_run:
        print('\nPlanned trials:')
        for tau, gamma in combos:
            print(f'  tau={tau}, gamma={gamma}, run_tag={build_run_tag(args.tag_prefix, tau, gamma)}')
        return

    rows: List[Dict[str, object]] = []
    for index, (tau, gamma) in enumerate(combos, start=1):
        run_tag = build_run_tag(args.tag_prefix, tau, gamma)
        print(f'\n[{index}/{len(combos)}] Launching tau={tau}, gamma={gamma} | run_tag={run_tag}')
        env = build_env(args, tau, gamma, run_tag)
        row = run_one_trial(tau=tau, gamma=gamma, env=env, run_tag=run_tag, run_log_dir=run_log_dir)
        rows.append(row)

        status = str(row.get('status', 'failed')).lower()
        if status == 'success':
            print(
                'Trial result | '
                f"best_valid_macro_f1={format_metric(row.get('best_valid_macro_f1'))} | "
                f"test_macro_f1={format_metric(row.get('test_macro_f1'))} | "
                f"test_qwk={format_metric(row.get('test_qwk'))} | "
                f"test_mae={format_metric(row.get('test_mae'))}"
            )
        else:
            print(f"Trial failed | {row.get('error', 'unknown error')}")

        write_results(rows, output_path)

    print_ranked_results(rows)
    print(f'\nSaved tuning results to: {output_path}')

    success_count = len([row for row in rows if str(row.get('status', '')).lower() == 'success'])
    fail_count = len(rows) - success_count
    print(f'Successful trials: {success_count} | Failed trials: {fail_count}')

    if success_count == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()