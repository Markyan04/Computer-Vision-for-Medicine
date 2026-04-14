#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate a chest X-ray checkpoint on the test split."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from chest_xray_loss_experiment_common import (  # noqa: E402
    DEFAULT_LOSS_ORDER,
    DEFAULT_TOPK,
    SEED,
    build_chestxray_dataloaders,
    count_parameters,
    create_medical_loss,
    evaluate,
    load_checkpoint_states,
    set_seed,
)
from checkpoint_eval_shared import (  # noqa: E402
    discover_model_scripts,
    format_top_metrics,
    infer_checkpoint_details,
    infer_dast_hparams_from_text,
    load_model_builder,
    normalize_script_path,
    print_confusion_matrix,
    resolve_device,
    resolve_output_path,
    save_confusion_matrix,
    save_summary,
)
from medical_losses import DistanceAwareSoftTargetLoss  # noqa: E402


LOG_DIR = THIS_DIR / 'logs'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a chest X-ray checkpoint on the test split.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path to evaluate.')
    parser.add_argument('--model', default='', help='Optional model script name or path. If omitted, the script tries to infer it from the checkpoint name.')
    parser.add_argument('--loss', default='', help='Optional loss name override. If omitted, the script tries to infer it from the checkpoint name.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Inference device. Default: auto.')
    parser.add_argument('--batch-size', type=int, default=int(os.getenv('CHESTXRAY_BATCH_SIZE', '32')))
    parser.add_argument('--num-workers', type=int, default=int(os.getenv('CHESTXRAY_NUM_WORKERS', '2')))
    parser.add_argument('--image-size', type=int, default=int(os.getenv('CHESTXRAY_IMAGE_SIZE', '224')))
    parser.add_argument('--val-ratio', type=float, default=float(os.getenv('CHESTXRAY_VAL_RATIO', '0.10')))
    parser.add_argument('--data-root', default='', help='Optional chest X-ray data directory override. Defaults to CHESTXRAY_DATA_ROOT or PROJECT_ROOT/CPN.')
    parser.add_argument('--train-dir', default='', help='Optional explicit train directory override.')
    parser.add_argument('--test-dir', default='', help='Optional explicit test directory override.')
    parser.add_argument('--dast-tau', type=float, default=None, help='Optional DAST tau override for loss reconstruction.')
    parser.add_argument('--dast-gamma', type=float, default=None, help='Optional DAST gamma override for loss reconstruction.')
    parser.add_argument('--output', default='', help='Optional CSV output path. Defaults to logs/checkpoint_eval_<timestamp>.csv.')
    parser.add_argument('--list-models', action='store_true', help='List available model scripts and exit.')
    return parser.parse_args()


def create_eval_loss(
    loss_name: str,
    num_classes: int,
    class_counts,
    feat_dim: int,
    device,
    dast_tau: Optional[float],
    dast_gamma: Optional[float],
):
    if loss_name == 'dast':
        tau = 1.0 if dast_tau is None else float(dast_tau)
        gamma = 1.5 if dast_gamma is None else float(dast_gamma)
        return DistanceAwareSoftTargetLoss(num_classes=num_classes, tau=tau, gamma=gamma).to(device)
    return create_medical_loss(
        loss_name=loss_name,
        num_classes=num_classes,
        class_counts=class_counts,
        feat_dim=feat_dim,
        device=device,
    )


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
) -> Dict[str, object]:
    row = {
        'checkpoint_path': str(checkpoint_path),
        'model_script': str(model_script),
        'loss_name': loss_name,
        'run_tag': run_tag,
        'dast_tau': dast_tau,
        'dast_gamma': dast_gamma,
        'device': str(device),
        'num_classes': data_bundle.num_classes,
        'test_size': data_bundle.split_sizes['test'],
        'test_loss': test_loss,
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
    for key, value in test_top.items():
        row[f'test_{key}'] = value
    return row


def main() -> None:
    args = parse_args()
    model_scripts = discover_model_scripts(THIS_DIR)

    if args.list_models:
        print('Available chest X-ray model scripts:')
        for name in model_scripts:
            print(f'  {name}')
        return

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    inferred_model, inferred_loss, inferred_run_tag = infer_checkpoint_details(
        checkpoint_path,
        model_scripts,
        DEFAULT_LOSS_ORDER,
    )

    if args.model:
        model_script_path = normalize_script_path(args.model, THIS_DIR)
    elif inferred_model is not None:
        model_script_path = normalize_script_path(inferred_model, THIS_DIR)
    else:
        raise ValueError('Unable to infer model script from checkpoint name. Please pass --model, for example --model ResNet_baseline.py')

    loss_name = (args.loss or inferred_loss or '').strip().lower()
    if not loss_name:
        raise ValueError('Unable to infer loss name from checkpoint name. Please pass --loss, for example --loss ce')
    if loss_name not in DEFAULT_LOSS_ORDER:
        raise ValueError(f'Unsupported loss name: {loss_name}')

    run_tag = inferred_run_tag
    inferred_tau, inferred_gamma = infer_dast_hparams_from_text(checkpoint_path.stem)
    dast_tau = args.dast_tau if args.dast_tau is not None else inferred_tau
    dast_gamma = args.dast_gamma if args.dast_gamma is not None else inferred_gamma

    data_root = Path(args.data_root or os.getenv('CHESTXRAY_DATA_ROOT', str(PROJECT_ROOT / 'CPN'))).resolve()
    train_dir = Path(args.train_dir or os.getenv('CHESTXRAY_TRAIN_DIR', str(data_root / 'train'))).resolve()
    test_dir = Path(args.test_dir or os.getenv('CHESTXRAY_TEST_DIR', str(data_root / 'test'))).resolve()
    device = resolve_device(args.device)
    output_path = resolve_output_path(args.output, LOG_DIR, checkpoint_path)

    set_seed(SEED)
    data_bundle = build_chestxray_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=SEED,
    )

    class_counts = np.bincount(data_bundle.train_targets, minlength=data_bundle.num_classes).tolist()
    build_model = load_model_builder(model_script_path)
    model = build_model(data_bundle.num_classes).to(device)
    feat_dim = model.fc.in_features
    criterion = create_eval_loss(
        loss_name=loss_name,
        num_classes=data_bundle.num_classes,
        class_counts=class_counts,
        feat_dim=feat_dim,
        device=device,
        dast_tau=dast_tau,
        dast_gamma=dast_gamma,
    )
    load_checkpoint_states(checkpoint_path, model, device, criterion=criterion)

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
    )
    save_summary(row, output_path)
    matrix_path = save_confusion_matrix(test_metrics['confusion_matrix'], data_bundle.class_names, output_path)

    print(f'Device: {device}')
    if str(device) == 'cuda':
        import torch
        print(f'CUDA: {torch.cuda.get_device_name(0)}')
    print(f'Data root: {data_root}')
    print(f'Train dir: {train_dir}')
    print(f'Test dir: {test_dir}')
    print(f'Model script: {model_script_path.name}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Loss: {loss_name}')
    if run_tag:
        print(f'Run tag: {run_tag}')
    if loss_name == 'dast':
        print(f'DAST hparams: tau={dast_tau if dast_tau is not None else 1.0}, gamma={dast_gamma if dast_gamma is not None else 1.5}')
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
        f'  loss={test_loss:.4f} | {format_top_metrics(test_top)} | '
        f"acc={test_metrics['acc'] * 100:.2f}% | bal_acc={test_metrics['balanced_acc'] * 100:.2f}% | "
        f"macro_f1={test_metrics['macro_f1']:.4f} | weighted_f1={test_metrics['weighted_f1']:.4f} | "
        f"precision_macro={test_metrics['precision_macro']:.4f} | recall_macro={test_metrics['recall_macro']:.4f} | "
        f"qwk={test_metrics['qwk']:.4f} | mae={test_metrics['mae']:.4f}"
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
    print(f'Saved confusion matrix CSV: {matrix_path}')


if __name__ == '__main__':
    main()
