#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Batch runner for Brain_Tumor_MRI_Loss experiments."""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
LOG_DIR = THIS_DIR / 'batch_logs'
LOSS_ENV_NAME = 'BRAIN_MRI_LOSSES'

DEFAULT_SCRIPTS = [
    'ResNet_baseline.py',
    'ResNet_layer2+MDFA.py',
    'ResNet_layer3+MDFA.py',
    'ResNet_layer2+GCSA.py',
    'ResNet_layer3+GCSA.py',
    'ResNet_layer2+MECS.py',
    'ResNet_layer3+MECS.py',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run one or more Brain_Tumor_MRI_Loss experiment scripts.')
    parser.add_argument('scripts', nargs='*', help='Optional script names to run. Defaults to all built-in scripts.')
    parser.add_argument('--losses', default='', help="Comma-separated loss names to run, e.g. 'pcol' or 'sce,gce,aom'.")
    return parser.parse_args()


def stream_run(script_name: str, losses: str = '') -> int:
    script_path = THIS_DIR / script_name
    if not script_path.exists():
        print(f'[SKIP] script not found: {script_path}')
        return 127

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'{script_path.stem}_{ts}.log'

    header = (
        f"\n{'=' * 100}\n"
        f'RUN SCRIPT: {script_name}\n'
        f"LOSSES: {losses or 'default'}\n"
        f"START TIME: {datetime.now().isoformat(timespec='seconds')}\n"
        f'LOG FILE: {log_path}\n'
        f"{'=' * 100}\n"
    )
    print(header, end='')

    env = os.environ.copy()
    if losses:
        env[LOSS_ENV_NAME] = losses
    else:
        env.pop(LOSS_ENV_NAME, None)
    cmd = [sys.executable, script_name]

    with open(log_path, 'w', encoding='utf-8') as fp:
        fp.write(header)
        fp.flush()

        proc = subprocess.Popen(
            cmd,
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
    scripts = args.scripts if args.scripts else DEFAULT_SCRIPTS
    losses = args.losses.strip()

    print(f"Batch start: {datetime.now().isoformat(timespec='seconds')}")
    print(f'Working dir: {THIS_DIR}')
    print(f'Scripts ({len(scripts)}): {scripts}')
    print(f"Losses: {losses or 'default'}")

    results = []
    for script in scripts:
        code = stream_run(script, losses=losses)
        results.append((script, code))

    print('\nBatch summary')
    print('-' * 100)
    for script, code in results:
        status = 'OK' if code == 0 else 'FAIL'
        print(f'{status:4} | code={code:3d} | {script}')
    print('-' * 100)
    print(f"Batch end: {datetime.now().isoformat(timespec='seconds')}")
    print(f'Batch logs dir: {LOG_DIR}')

    if any(code != 0 for _, code in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
