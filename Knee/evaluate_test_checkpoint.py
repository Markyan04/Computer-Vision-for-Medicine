#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate a Knee OA checkpoint on the test split."""

import argparse
import io
import os
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from checkpoint_eval_shared import (  # noqa: E402
    build_gaussian_noise_loader,
    discover_model_scripts,
    format_top_metrics,
    infer_dast_hparams_from_text,
    load_script_module,
    normalize_script_path,
    parse_float_list,
    print_confusion_matrix,
    resolve_device,
    resolve_output_path,
    save_rows,
    save_confusion_matrix,
    save_summary,
    save_text_report,
    slugify_float,
)
from medical_losses import DistanceAwareSoftTargetLoss  # noqa: E402


LOG_DIR = THIS_DIR / 'logs'
DEFAULT_CHECKPOINTS = {
    'ResNet_baseline.py': 'best_resnet50_knee_oa_standard_ce.pt',
    'ResNet_baseline+Loss4.py': 'best_resnet50_knee_oa_DAST.pt',
    'ResNet_baseline+WCE.py': 'best_resnet50_knee_oa_wce.pt',
    'ResNet_layer2+GCSA+CE.py': 'best_resnet50_gcsa_knee_oa.pt',
    'ResNet_layer2+GCSA+Loss4.py': 'best_resnet50_gcsa_dast_knee_oa.pt',
    'ResNet_layer3+GCSA+CE.py': 'best_resnet50_gcsa_layer3_knee_oa.pt',
    'ResNet_layer3+GCSA+Loss4.py': 'best_resnet50_gcsa_layer3_dast_knee_oa.pt',
    'ResNet_layer2+MDFA+CE.py': 'best_resnet50_mdfa_knee_oa.pt',
    'ResNet_layer2+MDFA+Loss4.py': 'best_resnet50_mdfa_dast_knee_oa.pt',
    'ResNet_layer3+MDFA+CE.py': 'best_resnet50_mdfa_layer3_knee_oa.pt',
    'ResNet_layer3+MDFA+Loss4.py': 'best_resnet50_mdfa_layer3_dast_knee_oa.pt',
    'ResNet_layer2+MECS+CE.py': 'best_resnet50_mecs_layer2_knee_oa.pt',
    'ResNet_layer2+MECS+Loss4.py': 'best_resnet50_mecs_layer2_dast_knee_oa.pt',
    'ResNet_layer3+MECS+CE.py': 'best_resnet50_mecs_layer3_knee_oa.pt',
    'ResNet_layer3+MECS+Loss4.py': 'best_resnet50_mecs_layer3_dast_knee_oa.pt',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a Knee OA checkpoint on the test split.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path to evaluate.')
    parser.add_argument('--model', default='', help='Optional model script name or path. If omitted, the script tries to infer it from the checkpoint filename.')
    parser.add_argument('--loss', default='', choices=['', 'ce', 'dast', 'wce'], help='Optional loss name override.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Inference device. Default: auto.')
    parser.add_argument('--batch-size', type=int, default=int(os.getenv('KNEE_BATCH_SIZE', '32')))
    parser.add_argument('--num-workers', type=int, default=int(os.getenv('KNEE_NUM_WORKERS', '4')))
    parser.add_argument('--image-size', type=int, default=int(os.getenv('KNEE_IMAGE_SIZE', '224')))
    parser.add_argument('--data-root', default='', help='Optional Knee data directory override. Defaults to KNEE_DATA_ROOT or PROJECT_ROOT/Knee_Osteoarthritis.')
    parser.add_argument('--dast-tau', type=float, default=None, help='Optional DAST tau override for loss reconstruction.')
    parser.add_argument('--dast-gamma', type=float, default=None, help='Optional DAST gamma override for loss reconstruction.')
    parser.add_argument('--output', default='', help='Optional CSV output path. Defaults to logs/checkpoint_eval_<timestamp>.csv.')
    parser.add_argument(
        '--gaussian-noise-stds',
        default='',
        help='Optional comma-separated Gaussian noise std list in pixel space, for example "0,0.05,0.1,0.15,0.2".',
    )
    parser.add_argument(
        '--gaussian-noise-seed',
        type=int,
        default=1234,
        help='Deterministic seed used for Gaussian noise generation. Default: 1234.',
    )
    parser.add_argument('--list-models', action='store_true', help='List available model scripts and exit.')
    return parser.parse_args()


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_model_script(checkpoint_path: Path) -> Optional[str]:
    target_name = checkpoint_path.name.lower()
    for script_name, checkpoint_name in DEFAULT_CHECKPOINTS.items():
        if checkpoint_name.lower() == target_name:
            return script_name
    return None


def infer_loss_name(script_name: str, checkpoint_path: Path, explicit_loss: str) -> str:
    if explicit_loss:
        return explicit_loss.lower()
    lower_script = script_name.lower()
    lower_checkpoint = checkpoint_path.name.lower()
    if 'wce' in lower_script or 'wce' in lower_checkpoint:
        return 'wce'
    if 'loss4' in lower_script or 'dast' in lower_checkpoint:
        return 'dast'
    return 'ce'


def build_model_from_module(module, num_classes: int):
    module_models = getattr(module, 'models', None)

    def instantiate(builder):
        if module_models is None or not hasattr(module_models, 'resnet50'):
            return builder()
        original_resnet50 = module_models.resnet50

        def safe_resnet50(*args, **kwargs):
            kwargs.pop('pretrained', None)
            kwargs['weights'] = None
            return original_resnet50(*args, **kwargs)

        module_models.resnet50 = safe_resnet50
        try:
            return builder()
        finally:
            module_models.resnet50 = original_resnet50

    if hasattr(module, 'build_model'):
        return instantiate(lambda: module.build_model(num_classes))
    for class_name in ('CustomResNet50MECS', 'CustomResNet50GCSA', 'CustomResNet50MDFA'):
        cls = getattr(module, class_name, None)
        if cls is not None:
            return instantiate(lambda cls=cls: cls(num_classes=num_classes))

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_knee_checkpoint_states(checkpoint_path: Path, model: torch.nn.Module, device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            return
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            return
    model.load_state_dict(checkpoint)


def create_eval_loss(loss_name: str, num_classes: int, train_targets, device, dast_tau: Optional[float], dast_gamma: Optional[float]):
    if loss_name == 'dast':
        tau = 1.0 if dast_tau is None else float(dast_tau)
        gamma = 1.5 if dast_gamma is None else float(dast_gamma)
        return DistanceAwareSoftTargetLoss(num_classes=num_classes, tau=tau, gamma=gamma).to(device)
    if loss_name == 'wce':
        class_counts = np.bincount(np.asarray(train_targets, dtype=np.int64), minlength=num_classes)
        total_samples = int(class_counts.sum())
        class_weights = np.zeros(num_classes, dtype=np.float32)
        nonzero_mask = class_counts > 0
        class_weights[nonzero_mask] = total_samples / (num_classes * class_counts[nonzero_mask])
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights_tensor)
    return nn.CrossEntropyLoss().to(device)


def count_parameters(module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def build_summary_row(
    checkpoint_path: Path,
    model_script: Path,
    loss_name: str,
    class_names,
    split_sizes,
    test_loss: float,
    test_top: Dict[str, float],
    test_metrics: Dict[str, object],
    dast_tau: Optional[float],
    dast_gamma: Optional[float],
    device,
    gaussian_noise_std: Optional[float] = None,
    gaussian_noise_seed: Optional[int] = None,
):
    row = {
        'checkpoint_path': str(checkpoint_path),
        'model_script': str(model_script),
        'loss_name': loss_name,
        'dast_tau': dast_tau,
        'dast_gamma': dast_gamma,
        'device': str(device),
        'gaussian_noise_std': gaussian_noise_std,
        'gaussian_noise_seed': gaussian_noise_seed,
        'num_classes': len(class_names),
        'test_size': split_sizes['test'],
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
        print('Available Knee model scripts:')
        for name in model_scripts:
            print(f'  {name}')
        return

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    if args.model:
        model_script_path = normalize_script_path(args.model, THIS_DIR)
    else:
        inferred_model = infer_model_script(checkpoint_path)
        if inferred_model is None:
            raise ValueError('Unable to infer model script from checkpoint filename. Please pass --model, for example --model ResNet_layer3+MECS+Loss4.py')
        model_script_path = normalize_script_path(inferred_model, THIS_DIR)

    loss_name = infer_loss_name(model_script_path.name, checkpoint_path, args.loss)
    if loss_name not in {'ce', 'dast', 'wce'}:
        raise ValueError(f'Unsupported loss name: {loss_name}')

    inferred_tau, inferred_gamma = infer_dast_hparams_from_text(checkpoint_path.stem)
    dast_tau = args.dast_tau if args.dast_tau is not None else inferred_tau
    dast_gamma = args.dast_gamma if args.dast_gamma is not None else inferred_gamma

    data_root = Path(args.data_root or os.getenv('KNEE_DATA_ROOT', str(PROJECT_ROOT / 'Knee_Osteoarthritis'))).resolve()
    output_path = resolve_output_path(args.output, LOG_DIR, checkpoint_path)
    device = resolve_device(args.device)
    noise_stds = parse_float_list(args.gaussian_noise_stds)
    if noise_stds and not args.output:
        output_path = output_path.with_name(output_path.name.replace('checkpoint_eval_', 'gaussian_noise_eval_', 1))

    module = load_script_module(model_script_path, prefix='knee_eval')
    seed = int(getattr(module, 'SEED', 1234))
    set_seed(seed)
    module.DATA_ROOT = str(data_root)
    module.BATCH_SIZE = args.batch_size
    module.NUM_WORKERS = args.num_workers
    module.IMG_SIZE = args.image_size

    loader_stdout = io.StringIO()
    with redirect_stdout(loader_stdout):
        train_loader, valid_loader, test_loader, auto_test_loader, train_dataset = module.make_dataloaders()

    class_names = list(getattr(module, 'CLASS_NAMES', train_dataset.classes))
    topk = tuple(getattr(module, 'TOPK', (1, 2, 3)))
    split_sizes = {
        'train': len(train_loader.dataset),
        'valid': len(valid_loader.dataset),
        'test': len(test_loader.dataset),
        'auto_test': len(auto_test_loader.dataset) if auto_test_loader is not None else 0,
    }

    model = build_model_from_module(module, num_classes=len(class_names)).to(device)
    criterion = create_eval_loss(
        loss_name=loss_name,
        num_classes=len(class_names),
        train_targets=train_dataset.targets,
        device=device,
        dast_tau=dast_tau,
        dast_gamma=dast_gamma,
    )
    load_knee_checkpoint_states(checkpoint_path, model, device)

    if noise_stds:
        rows = []
        log_lines = [
            'Gaussian Noise Robustness Analysis',
            f'Device: {device}',
            f'Data root: {data_root}',
            f'Model script: {model_script_path.name}',
            f'Checkpoint: {checkpoint_path}',
            f'Loss: {loss_name}',
            f'Noise stds: {", ".join(f"{value:.4f}" for value in noise_stds)}',
            f'Noise seed: {args.gaussian_noise_seed}',
        ]
        if loss_name == 'dast':
            log_lines.append(f'DAST hparams: tau={dast_tau if dast_tau is not None else 1.0}, gamma={dast_gamma if dast_gamma is not None else 1.5}')
        log_lines.extend([
            f'Classes ({len(class_names)}): {class_names}',
            (
                f"Split sizes | train={split_sizes['train']}, valid={split_sizes['valid']}, test={split_sizes['test']}"
                + (f", auto_test={split_sizes['auto_test']}" if split_sizes['auto_test'] else '')
            ),
            f'Model params: {count_parameters(model):,}',
            f'Criterion: {criterion.__class__.__name__} | trainable_params={count_parameters(criterion):,}',
            '',
        ])

        print(f'Device: {device}')
        if str(device) == 'cuda':
            print(f'CUDA: {torch.cuda.get_device_name(0)}')
        print(f'Data root: {data_root}')
        print(f'Gaussian noise robustness analysis for stds: {[round(value, 4) for value in noise_stds]}')

        for noise_std in noise_stds:
            eval_loader = test_loader
            if noise_std > 0:
                eval_loader = build_gaussian_noise_loader(
                    test_loader,
                    noise_std=noise_std,
                    seed=args.gaussian_noise_seed,
                )

            test_loss, test_top, test_metrics = module.evaluate(
                model=model,
                loader=eval_loader,
                criterion=criterion,
                device=device,
                topk=topk,
            )

            row = build_summary_row(
                checkpoint_path=checkpoint_path,
                model_script=model_script_path,
                loss_name=loss_name,
                class_names=class_names,
                split_sizes=split_sizes,
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
            matrix_path = save_confusion_matrix(test_metrics['confusion_matrix'], class_names, per_std_output)
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
            if test_metrics.get('ovr_roc_auc_macro') is not None:
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

    test_loss, test_top, test_metrics = module.evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        topk=topk,
    )

    row = build_summary_row(
        checkpoint_path=checkpoint_path,
        model_script=model_script_path,
        loss_name=loss_name,
        class_names=class_names,
        split_sizes=split_sizes,
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
    matrix_path = save_confusion_matrix(test_metrics['confusion_matrix'], class_names, output_path)

    print(f'Device: {device}')
    if str(device) == 'cuda':
        print(f'CUDA: {torch.cuda.get_device_name(0)}')
    print(f'Data root: {data_root}')
    print(f'Model script: {model_script_path.name}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Loss: {loss_name}')
    if loss_name == 'dast':
        print(f'DAST hparams: tau={dast_tau if dast_tau is not None else 1.0}, gamma={dast_gamma if dast_gamma is not None else 1.5}')
    print(f'Classes ({len(class_names)}): {class_names}')
    print(
        f"Split sizes | train={split_sizes['train']}, valid={split_sizes['valid']}, test={split_sizes['test']}"
        + (f", auto_test={split_sizes['auto_test']}" if split_sizes['auto_test'] else '')
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
    if test_metrics.get('ovr_roc_auc_macro') is not None:
        print(
            f"  ovr_roc_auc_macro={test_metrics['ovr_roc_auc_macro']:.4f} | "
            f"ovr_pr_auc_macro={test_metrics['ovr_pr_auc_macro']:.4f}"
        )
    print('')
    print_confusion_matrix(test_metrics['confusion_matrix'], class_names)
    print('')
    print('Classification Report:')
    print(test_metrics['classification_report'])
    print('')
    print(f'Saved summary CSV: {output_path}')
    print(f'Saved confusion matrix CSV: {matrix_path}')


if __name__ == '__main__':
    main()

