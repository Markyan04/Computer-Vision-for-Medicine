#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared helpers for checkpoint evaluation scripts."""

import csv
import importlib.util
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def parse_float_list(raw_values: str) -> list[float]:
    values = []
    seen = set()
    for chunk in raw_values.split(','):
        token = chunk.strip()
        if not token:
            continue
        value = float(token)
        if value < 0:
            raise ValueError(f'Noise/std values must be non-negative, got {value}')
        key = f'{value:.10f}'
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return values


def slugify_float(value: float) -> str:
    text = f'{float(value):.6f}'.rstrip('0').rstrip('.')
    if not text:
        text = '0'
    return text.replace('-', 'm').replace('.', 'p')


def save_summary(row, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_rows(rows: Iterable[dict], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError('save_rows() requires at least one row.')

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_text_report(lines: Sequence[str], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines))
        if lines:
            fp.write('\n')
    return output_path


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


class GaussianNoiseDataset:
    def __init__(
        self,
        base_dataset,
        noise_std: float,
        seed: int,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ):
        self.base_dataset = base_dataset
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.mean = tuple(float(item) for item in mean)
        self.std = tuple(float(item) for item in std)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        import torch

        x, y = self.base_dataset[index]
        if self.noise_std <= 0:
            return x, y
        if not torch.is_tensor(x):
            raise TypeError('GaussianNoiseDataset expects the base dataset to return tensors.')

        mean = torch.tensor(self.mean, dtype=x.dtype).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=x.dtype).view(-1, 1, 1)
        if mean.size(0) != x.size(0):
            mean = mean[:1].expand(x.size(0), -1, -1)
            std = std[:1].expand(x.size(0), -1, -1)

        generator = torch.Generator()
        generator.manual_seed(self.seed + int(index))
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)

        x_01 = torch.clamp(x * std + mean, 0.0, 1.0)
        x_01 = torch.clamp(x_01 + noise * self.noise_std, 0.0, 1.0)
        x = (x_01 - mean) / std
        return x, y

    def __getattr__(self, name):
        # On Windows, DataLoader workers use spawn and may look up attributes
        # while the dataset object is still being unpickled. Guard against
        # recursive fallback before base_dataset is restored.
        base_dataset = self.__dict__.get('base_dataset')
        if base_dataset is None:
            raise AttributeError(name)
        return getattr(base_dataset, name)


def build_gaussian_noise_loader(
    loader,
    noise_std: float,
    seed: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
):
    import torch.utils.data as data

    dataset = GaussianNoiseDataset(
        base_dataset=loader.dataset,
        noise_std=noise_std,
        seed=seed,
        mean=mean,
        std=std,
    )

    kwargs = {
        'batch_size': loader.batch_size,
        'shuffle': False,
        'num_workers': loader.num_workers,
        'collate_fn': loader.collate_fn,
        'pin_memory': loader.pin_memory,
        'drop_last': loader.drop_last,
    }

    worker_init_fn = getattr(loader, 'worker_init_fn', None)
    if worker_init_fn is not None:
        kwargs['worker_init_fn'] = worker_init_fn

    timeout = getattr(loader, 'timeout', 0)
    if timeout:
        kwargs['timeout'] = timeout

    generator = getattr(loader, 'generator', None)
    if generator is not None:
        kwargs['generator'] = generator

    multiprocessing_context = getattr(loader, 'multiprocessing_context', None)
    if multiprocessing_context is not None:
        kwargs['multiprocessing_context'] = multiprocessing_context

    if loader.num_workers > 0:
        kwargs['persistent_workers'] = getattr(loader, 'persistent_workers', False)
        prefetch_factor = getattr(loader, 'prefetch_factor', None)
        if prefetch_factor is not None:
            kwargs['prefetch_factor'] = prefetch_factor

    pin_memory_device = getattr(loader, 'pin_memory_device', '')
    if pin_memory_device:
        kwargs['pin_memory_device'] = pin_memory_device

    return data.DataLoader(dataset, **kwargs)
