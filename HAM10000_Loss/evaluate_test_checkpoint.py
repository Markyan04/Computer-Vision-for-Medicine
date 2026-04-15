#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate a HAM10000 checkpoint on the test split."""

import argparse
import csv
import importlib.util
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from ham10000_loss_experiment_common import (  # noqa: E402
    DEFAULT_LOSS_ORDER,
    SEED,
    build_ham10000_dataloaders,
    count_parameters,
    create_medical_loss,
    evaluate,
    has_trainable_parameters,
    load_checkpoint_states,
    set_seed,
)
from checkpoint_eval_shared import (  # noqa: E402
    build_gaussian_noise_loader,
    format_top_metrics,
    parse_float_list,
    save_rows,
    save_text_report,
    slugify_float,
)


LOG_DIR = THIS_DIR / 'logs'


def discover_model_scripts() -> List[str]:
    scripts = []
    for path in sorted(THIS_DIR.glob('ResNet*.py')):
        if path.name.endswith('_warmstart.py'):
            continue
        scripts.append(path.name)
    return scripts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a HAM10000 checkpoint on the test split.')
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Checkpoint path to evaluate.',
    )
    parser.add_argument(
        '--model',
        default='',
        help='Optional model script name or path. If omitted, the script tries to infer it from the checkpoint name.',
    )
    parser.add_argument(
        '--loss',
        default='',
        help='Optional loss name override. If omitted, the script tries to infer it from the checkpoint name.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Inference device. Default: auto.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=int(os.getenv('HAM10000_BATCH_SIZE', '32')),
        help='Evaluation batch size. Default reads HAM10000_BATCH_SIZE or falls back to 32.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=int(os.getenv('HAM10000_NUM_WORKERS', '2')),
        help='Evaluation num_workers. Default reads HAM10000_NUM_WORKERS or falls back to 2.',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=int(os.getenv('HAM10000_IMAGE_SIZE', '224')),
        help='Evaluation image size. Default reads HAM10000_IMAGE_SIZE or falls back to 224.',
    )
    parser.add_argument(
        '--data-dir',
        default='',
        help='Optional HAM10000 data directory override. Defaults to HAM10000_DATA_DIR or PROJECT_ROOT/ISIC.',
    )
    parser.add_argument(
        '--dast-tau',
        type=float,
        default=None,
        help='Optional DAST tau override for loss reconstruction.',
    )
    parser.add_argument(
        '--dast-gamma',
        type=float,
        default=None,
        help='Optional DAST gamma override for loss reconstruction.',
    )
    parser.add_argument(
        '--output',
        default='',
        help='Optional CSV output path. Defaults to logs/checkpoint_eval_<timestamp>.csv.',
    )
    parser.add_argument(
        '--gaussian-noise-stds',
        default='',
        help='Optional comma-separated Gaussian noise std list in pixel space, for example "0,0.05,0.1,0.15,0.2".',
    )
    parser.add_argument(
        '--gaussian-noise-seed',
        type=int,
        default=SEED,
        help='Deterministic seed used for Gaussian noise generation. Default: experiment seed.',
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available model scripts and exit.',
    )
    return parser.parse_args()


def resolve_device(device_name: str):
    import torch

    if device_name == 'cpu':
        return torch.device('cpu')
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_builder(script_path: Path):
    module_name = 'ham_eval_' + re.sub(r'[^0-9A-Za-z_]+', '_', script_path.stem)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load model script: {script_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_model = getattr(module, 'build_model', None)
    if build_model is None:
        raise AttributeError(f'build_model() not found in: {script_path}')
    return build_model


def normalize_script_path(raw_model: str) -> Path:
    candidate = Path(raw_model)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    local_candidate = (THIS_DIR / raw_model).resolve()
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
    for loss_name in sorted(DEFAULT_LOSS_ORDER, key=len, reverse=True):
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


def resolve_output_path(raw_output: str, checkpoint_path: Path) -> Path:
    if raw_output:
        return Path(raw_output).expanduser().resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', checkpoint_path.stem).strip('._') or 'checkpoint'
    return (LOG_DIR / f'checkpoint_eval_{stem}_{timestamp}.csv').resolve()


def build_summary_row(
    checkpoint_path: Path,
    model_script: Path,
    loss_name: str,
    run_tag: str,
    data_bundle,
    test_loss: float,
    test_top: Dict[str, float],
    test_metrics: Dict[str, object],
    dast_tau: Optional[float],
    dast_gamma: Optional[float],
    device,
    gaussian_noise_std: Optional[float] = None,
    gaussian_noise_seed: Optional[int] = None,
) -> Dict[str, object]:
    return {
        'checkpoint_path': str(checkpoint_path),
        'model_script': str(model_script),
        'loss_name': loss_name,
        'run_tag': run_tag,
        'dast_tau': dast_tau,
        'dast_gamma': dast_gamma,
        'device': str(device),
        'gaussian_noise_std': gaussian_noise_std,
        'gaussian_noise_seed': gaussian_noise_seed,
        'num_classes': data_bundle.num_classes,
        'test_size': data_bundle.split_sizes['test'],
        'test_loss': test_loss,
        'test_top1': test_top.get('top1'),
        'test_top3': test_top.get('top3'),
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
    }


def save_summary(row: Dict[str, object], output_path: Path) -> None:
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
    header = 'true\\pred'.ljust(10) + ' ' + ' '.join(name.rjust(6) for name in class_names)
    print(header)
    for class_name, row in zip(class_names, confusion_matrix.tolist()):
        values = ' '.join(f'{int(value):6d}' for value in row)
        print(f'{class_name:<10} {values}')


def main() -> None:
    args = parse_args()
    model_scripts = discover_model_scripts()

    if args.list_models:
        print('Available HAM10000 model scripts:')
        for name in model_scripts:
            print(f'  {name}')
        return

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    inferred_model, inferred_loss, inferred_run_tag = infer_checkpoint_details(checkpoint_path, model_scripts)

    model_script_path: Optional[Path] = None
    if args.model:
        model_script_path = normalize_script_path(args.model)
    elif inferred_model is not None:
        model_script_path = normalize_script_path(inferred_model)
    else:
        raise ValueError(
            'Unable to infer model script from checkpoint name. '
            'Please pass --model, for example --model ResNet_baseline.py'
        )

    loss_name = (args.loss or inferred_loss or '').strip().lower()
    if not loss_name:
        raise ValueError(
            'Unable to infer loss name from checkpoint name. '
            'Please pass --loss, for example --loss ce'
        )
    if loss_name not in DEFAULT_LOSS_ORDER:
        raise ValueError(f'Unsupported loss name: {loss_name}')

    run_tag = inferred_run_tag
    inferred_tau, inferred_gamma = infer_dast_hparams_from_text(checkpoint_path.stem)
    dast_tau = args.dast_tau if args.dast_tau is not None else inferred_tau
    dast_gamma = args.dast_gamma if args.dast_gamma is not None else inferred_gamma

    if loss_name == 'dast':
        if dast_tau is not None:
            os.environ['HAM10000_DAST_TAU'] = str(dast_tau)
        if dast_gamma is not None:
            os.environ['HAM10000_DAST_GAMMA'] = str(dast_gamma)

    data_dir = Path(args.data_dir or os.getenv('HAM10000_DATA_DIR', str(PROJECT_ROOT / 'ISIC'))).resolve()
    device = resolve_device(args.device)
    output_path = resolve_output_path(args.output, checkpoint_path)
    noise_stds = parse_float_list(args.gaussian_noise_stds)
    if noise_stds and not args.output:
        output_path = output_path.with_name(output_path.name.replace('checkpoint_eval_', 'gaussian_noise_eval_', 1))

    set_seed(SEED)
    data_bundle = build_ham10000_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=SEED,
        data_dir=data_dir,
    )

    import numpy as np

    class_counts = np.bincount(data_bundle.train_targets, minlength=data_bundle.num_classes).tolist()
    build_model = load_model_builder(model_script_path)
    model = build_model(data_bundle.num_classes).to(device)
    feat_dim = model.fc.in_features
    criterion = create_medical_loss(
        loss_name=loss_name,
        num_classes=data_bundle.num_classes,
        class_counts=class_counts,
        feat_dim=feat_dim,
        device=device,
    )
    load_checkpoint_states(checkpoint_path, model, device, criterion=criterion)

    if noise_stds:
        rows = []
        log_lines = [
            'Gaussian Noise Robustness Analysis',
            f'Device: {device}',
            f'Data dir: {data_dir}',
            f'Model script: {model_script_path.name}',
            f'Checkpoint: {checkpoint_path}',
            f'Loss: {loss_name}',
            f'Noise stds: {", ".join(f"{value:.4f}" for value in noise_stds)}',
            f'Noise seed: {args.gaussian_noise_seed}',
        ]
        if run_tag:
            log_lines.append(f'Run tag: {run_tag}')
        if loss_name == 'dast':
            log_lines.append(f'DAST hparams: tau={dast_tau if dast_tau is not None else "default"}, gamma={dast_gamma if dast_gamma is not None else "default"}')
        log_lines.extend([
            f'Classes ({data_bundle.num_classes}): {data_bundle.class_names}',
            (
                f"Split sizes | train={data_bundle.split_sizes['train']}, "
                f"valid={data_bundle.split_sizes['valid']}, test={data_bundle.split_sizes['test']}"
            ),
            f'Model params: {count_parameters(model):,}',
            f'Criterion: {criterion.__class__.__name__} | trainable_params={count_parameters(criterion):,}',
            '',
        ])

        print(f'Device: {device}')
        if str(device) == 'cuda':
            import torch
            print(f'CUDA: {torch.cuda.get_device_name(0)}')
        print(f'Data dir: {data_dir}')
        print(f'Gaussian noise robustness analysis for stds: {[round(value, 4) for value in noise_stds]}')

        for noise_std in noise_stds:
            eval_loader = data_bundle.test_loader
            if noise_std > 0:
                eval_loader = build_gaussian_noise_loader(
                    data_bundle.test_loader,
                    noise_std=noise_std,
                    seed=args.gaussian_noise_seed,
                )

            test_loss, test_top, test_metrics = evaluate(
                model=model,
                loader=eval_loader,
                criterion=criterion,
                device=device,
                loss_name=loss_name,
                num_classes=data_bundle.num_classes,
                class_names=data_bundle.class_names,
                topk=(1, 3),
            )

            row = build_summary_row(
                checkpoint_path=checkpoint_path,
                model_script=model_script_path,
                loss_name=loss_name,
                run_tag=run_tag,
                data_bundle=data_bundle,
                test_loss=test_loss,
                test_top=test_top,
                test_metrics=test_metrics,
                dast_tau=dast_tau,
                dast_gamma=dast_gamma,
                device=device,
                gaussian_noise_std=noise_std,
                gaussian_noise_seed=args.gaussian_noise_seed,
            )
            rows.append(row)

            per_std_output = output_path.with_name(
                f'{output_path.stem}_std{slugify_float(noise_std)}{output_path.suffix}'
            )
            matrix_path = save_confusion_matrix(test_metrics['confusion_matrix'], data_bundle.class_names, per_std_output)
            summary_line = (
                f'std={noise_std:.4f} | loss={test_loss:.4f} | {format_top_metrics(test_top)} | '
                f"acc={test_metrics['acc'] * 100:.2f}% | bal_acc={test_metrics['balanced_acc'] * 100:.2f}% | "
                f"macro_f1={test_metrics['macro_f1']:.4f} | weighted_f1={test_metrics['weighted_f1']:.4f} | "
                f"precision_macro={test_metrics['precision_macro']:.4f} | recall_macro={test_metrics['recall_macro']:.4f} | "
                f"qwk={test_metrics['qwk']:.4f} | mae={test_metrics['mae']:.4f}"
            )
            print(summary_line)
            log_lines.extend([
                f'[std={noise_std:.4f}]',
                summary_line,
            ])
            if test_metrics['ovr_roc_auc_macro'] is not None:
                auc_line = (
                    f"ovr_roc_auc_macro={test_metrics['ovr_roc_auc_macro']:.4f} | "
                    f"ovr_pr_auc_macro={test_metrics['ovr_pr_auc_macro']:.4f}"
                )
                print(f'  {auc_line}')
                log_lines.append(auc_line)
            log_lines.extend([
                'Classification Report:',
                str(test_metrics['classification_report']),
                f'Confusion matrix CSV: {matrix_path}',
                '',
            ])

        # save_rows(rows, output_path)
        log_path = save_text_report(log_lines, output_path.with_suffix('.log'))
        print('')
        # print(f'Saved robustness CSV: {output_path}')
        print(f'Saved robustness log: {log_path}')
        return

    test_loss, test_top, test_metrics = evaluate(
        model=model,
        loader=data_bundle.test_loader,
        criterion=criterion,
        device=device,
        loss_name=loss_name,
        num_classes=data_bundle.num_classes,
        class_names=data_bundle.class_names,
        topk=(1, 3),
    )

    row = build_summary_row(
        checkpoint_path=checkpoint_path,
        model_script=model_script_path,
        loss_name=loss_name,
        run_tag=run_tag,
        data_bundle=data_bundle,
        test_loss=test_loss,
        test_top=test_top,
        test_metrics=test_metrics,
        dast_tau=dast_tau,
        dast_gamma=dast_gamma,
        device=device,
        gaussian_noise_std=0.0,
        gaussian_noise_seed=None,
    )
    save_summary(row, output_path)
    save_confusion_matrix(test_metrics['confusion_matrix'], data_bundle.class_names, output_path)

    print(f'Device: {device}')
    if str(device) == 'cuda':
        import torch
        print(f'CUDA: {torch.cuda.get_device_name(0)}')
    print(f'Data dir: {data_dir}')
    print(f'Model script: {model_script_path.name}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Loss: {loss_name}')
    if run_tag:
        print(f'Run tag: {run_tag}')
    if loss_name == 'dast':
        print(f'DAST hparams: tau={dast_tau if dast_tau is not None else "default"}, gamma={dast_gamma if dast_gamma is not None else "default"}')
    print(f'Classes ({data_bundle.num_classes}): {data_bundle.class_names}')
    print(
        f"Split sizes | train={data_bundle.split_sizes['train']}, "
        f"valid={data_bundle.split_sizes['valid']}, test={data_bundle.split_sizes['test']}"
    )
    print(f'Model params: {count_parameters(model):,}')
    print(f'Criterion: {criterion.__class__.__name__} | trainable_params={count_parameters(criterion):,}')
    print('')
    print('[TEST]')
    print(
        f"  loss={test_loss:.4f} | {format_top_metrics(test_top)} | "
        f"acc={test_metrics['acc'] * 100:.2f}% | "
        f"bal_acc={test_metrics['balanced_acc'] * 100:.2f}% | macro_f1={test_metrics['macro_f1']:.4f} | "
        f"weighted_f1={test_metrics['weighted_f1']:.4f} | precision_macro={test_metrics['precision_macro']:.4f} | "
        f"recall_macro={test_metrics['recall_macro']:.4f} | qwk={test_metrics['qwk']:.4f} | mae={test_metrics['mae']:.4f}"
    )
    if test_metrics['ovr_roc_auc_macro'] is not None:
        print(
            f"  ovr_roc_auc_macro={test_metrics['ovr_roc_auc_macro']:.4f} | "
            f"ovr_pr_auc_macro={test_metrics['ovr_pr_auc_macro']:.4f}"
        )
    print('')
    print_confusion_matrix(test_metrics['confusion_matrix'], data_bundle.class_names)
    print('')
    print('Classification Report:')
    print(test_metrics['classification_report'])
    print('')
    print(f'Saved summary CSV: {output_path}')
    print(f'Saved confusion matrix CSV: {output_path.with_name(output_path.stem + "_confusion_matrix.csv")}')


if __name__ == '__main__':
    main()
