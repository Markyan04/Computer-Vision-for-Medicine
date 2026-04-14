#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared helpers for checkpoint evaluation scripts."""

import csv
import importlib.util
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple


def discover_model_scripts(directory: Path, pattern: str = 'ResNet*.py', exclude_suffixes: Sequence[str] = ()):
    scripts = []
    for path in sorted(directory.glob(pattern)):
        if any(path.name.endswith(suffix) for suffix in exclude_suffixes):
            continue
        scripts.append(path.name)
    return scripts


def resolve_device(device_name: str):
    import torch

    if device_name == 'cpu':
        return torch.device('cpu')
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_script_module(script_path: Path, prefix: str = 'checkpoint_eval'):
    module_name = prefix + '_' + re.sub(r'[^0-9A-Za-z_]+', '_', script_path.stem)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load model script: {script_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_builder(script_path: Path):
    module = load_script_module(script_path, prefix='checkpoint_builder')
    build_model = getattr(module, 'build_model', None)
    if build_model is None:
        raise AttributeError(f'build_model() not found in: {script_path}')
    return build_model


def normalize_script_path(raw_model: str, base_dir: Path) -> Path:
    candidate = Path(raw_model)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    local_candidate = (base_dir / raw_model).resolve()
    if local_candidate.exists():
        return local_candidate
    raise FileNotFoundError(f'Model script not found: {raw_model}')


def unslug_float(text: str) -> float:
    token = text.strip().lower().replace('p', '.')
    if token.startswith('m'):
        token = '-' + token[1:]
    return float(token)


def infer_dast_hparams_from_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    match = re.search(r'tau(?P<tau>m?\d+(?:p\d+)?)_gamma(?P<gamma>m?\d+(?:p\d+)?)', text)
    if match is None:
        return None, None
    return unslug_float(match.group('tau')), unslug_float(match.group('gamma'))


def infer_checkpoint_details(
    checkpoint_path: Path,
    model_scripts: Sequence[str],
    loss_names: Sequence[str],
) -> Tuple[Optional[str], Optional[str], str]:
    name = checkpoint_path.stem
    if name.startswith('best_'):
        name = name[5:]

    matched_script: Optional[str] = None
    remaining = name
    for script_name in sorted(model_scripts, key=lambda item: len(Path(item).stem), reverse=True):
        script_stem = Path(script_name).stem
        prefix = script_stem + '_'
        if name.startswith(prefix):
            matched_script = script_name
            remaining = name[len(prefix):]
            break

    matched_loss: Optional[str] = None
    run_tag = ''
    for loss_name in sorted(loss_names, key=len, reverse=True):
        if remaining == loss_name:
            matched_loss = loss_name
            run_tag = ''
            break
        prefix = loss_name + '_'
        if remaining.startswith(prefix):
            matched_loss = loss_name
            run_tag = remaining[len(prefix):]
            break

    return matched_script, matched_loss, run_tag


def resolve_output_path(raw_output: str, log_dir: Path, checkpoint_path: Path) -> Path:
    if raw_output:
        return Path(raw_output).expanduser().resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', checkpoint_path.stem).strip('._') or 'checkpoint'
    return (log_dir / f'checkpoint_eval_{stem}_{timestamp}.csv').resolve()


def save_summary(row, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_confusion_matrix(confusion_matrix, class_names: Sequence[str], output_path: Path) -> Path:
    matrix_path = output_path.with_name(output_path.stem + '_confusion_matrix.csv')
    with open(matrix_path, 'w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['true\\pred'] + list(class_names))
        for class_name, row in zip(class_names, confusion_matrix.tolist()):
            writer.writerow([class_name] + [int(value) for value in row])
    return matrix_path


def print_confusion_matrix(confusion_matrix, class_names: Sequence[str]) -> None:
    print('Confusion Matrix:')
    header = 'true\\pred'.ljust(10) + ' ' + ' '.join(name.rjust(10) for name in class_names)
    print(header)
    for class_name, row in zip(class_names, confusion_matrix.tolist()):
        values = ' '.join(f'{int(value):10d}' for value in row)
        print(f'{class_name:<10} {values}')


def format_top_metrics(test_top) -> str:
    def sort_key(item):
        match = re.search(r'(\d+)', item[0])
        return int(match.group(1)) if match else 0

    parts = []
    for key, value in sorted(test_top.items(), key=sort_key):
        parts.append(f'{key}={value * 100:.2f}%')
    return ' | '.join(parts)
