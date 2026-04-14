#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Draw Brain Tumor MRI Grad-CAM visualizations with pytorch-grad-cam."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from brain_tumor_mri_loss_experiment_common import load_checkpoint_states, sanitize_run_tag, set_seed, SEED  # noqa: E402
from gradcam_shared import compose_panel, extract_logits, load_script_module, resolve_device, sanitize_filename, tensor_to_pil  # noqa: E402
from pytorch_grad_cam_shared import (  # noqa: E402
    CAM_METHOD_CHOICES,
    build_cam_images,
    ensure_pytorch_grad_cam,
    predict,
    resolve_target_module,
)


CHECKPOINT_DIR = THIS_DIR / 'checkpoints'
OUTPUT_DIR = THIS_DIR / 'gradcam_outputs_pytorch_grad_cam'
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def discover_model_scripts() -> List[str]:
    return [path.name for path in sorted(THIS_DIR.glob('ResNet*.py'))]


def build_eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def resolve_test_dir(args: argparse.Namespace) -> Path:
    if args.test_dir:
        return Path(args.test_dir).expanduser().resolve()
    if getattr(args, 'data_root', ''):
        return (Path(args.data_root).expanduser().resolve() / 'Testing').resolve()

    env_test_dir = os.getenv('BRAIN_MRI_TEST_DIR', '').strip()
    if env_test_dir:
        return Path(env_test_dir).expanduser().resolve()

    env_data_root = os.getenv('BRAIN_MRI_DATA_ROOT', '').strip()
    if env_data_root:
        return (Path(env_data_root).expanduser().resolve() / 'Testing').resolve()

    return (PROJECT_ROOT / 'Brain_Tumor_MRI' / 'Testing').resolve()


def build_test_records(test_dir: Path) -> Dict[str, object]:
    dataset = ImageFolder(str(test_dir))
    rows = []
    for path, label in dataset.samples:
        rows.append({
            'image_path': path,
            'label_index': int(label),
            'label_name': dataset.classes[label],
            'relative_path': str(Path(path).relative_to(test_dir)),
        })
    return {
        'dataset': dataset,
        'class_names': list(dataset.classes),
        'rows': rows,
    }


def resolve_image_source(args: argparse.Namespace, test_dir: Path, records: Dict[str, object]) -> Dict[str, object]:
    rows = list(records['rows'])

    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f'Image path not found: {image_path}')
        return {
            'image_path': str(image_path),
            'display_name': image_path.stem,
            'true_label_name': None,
            'true_label_index': None,
        }

    if args.relative_path:
        image_path = (test_dir / args.relative_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f'Relative image path not found under test dir: {image_path}')
        for row in rows:
            if Path(row['image_path']).resolve() == image_path:
                return {
                    'image_path': row['image_path'],
                    'display_name': Path(row['relative_path']).stem,
                    'true_label_name': row['label_name'],
                    'true_label_index': row['label_index'],
                }
        return {
            'image_path': str(image_path),
            'display_name': image_path.stem,
            'true_label_name': None,
            'true_label_index': None,
        }

    if args.class_name:
        target_class = args.class_name.strip().lower()
        rows = [row for row in rows if str(row['label_name']).lower() == target_class]

    if not rows:
        raise RuntimeError('No matching samples found in the test set.')
    if args.sample_index < 0 or args.sample_index >= len(rows):
        raise IndexError(f'sample-index out of range: {args.sample_index} (size={len(rows)})')

    row = rows[args.sample_index]
    return {
        'image_path': row['image_path'],
        'display_name': Path(row['relative_path']).stem,
        'true_label_name': row['label_name'],
        'true_label_index': row['label_index'],
    }


def parse_args() -> argparse.Namespace:
    script_choices = discover_model_scripts()
    parser = argparse.ArgumentParser(description='Draw Brain Tumor MRI Grad-CAM with pytorch-grad-cam.')
    parser.add_argument('--model', default='ResNet_layer3+MECS.py', choices=script_choices)
    parser.add_argument('--loss', default='ce', help="Loss suffix used in checkpoint naming, e.g. 'ce' or 'dast'.")
    parser.add_argument('--run-tag', default='', help='Optional run tag used in checkpoint naming.')
    parser.add_argument('--checkpoint', default='', help='Optional explicit checkpoint path.')
    parser.add_argument('--target-layer', default='layer4', help='Layer/module path to visualize, e.g. layer4 or inserted_module.')
    parser.add_argument('--target-class', type=int, default=None, help='Optional target class index. Overrides --cam-on.')
    parser.add_argument('--cam-on', default='pred', choices=['pred', 'true'], help='Which class to explain when --target-class is not provided.')
    parser.add_argument('--cam-method', default='gradcam++', choices=CAM_METHOD_CHOICES, help='pytorch-grad-cam method. Default: gradcam++.')
    parser.add_argument('--aug-smooth', action='store_true', help='Enable test-time augmentation smoothing if supported.')
    parser.add_argument('--eigen-smooth', action='store_true', help='Enable eigen smoothing if supported.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--image-size', type=int, default=int(os.getenv('BRAIN_MRI_IMAGE_SIZE', '224')))
    parser.add_argument('--alpha', type=float, default=0.35, help='Heatmap opacity. Default: 0.35.')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    parser.add_argument('--data-root', default='', help='Optional BRAIN_MRI data root override.')
    parser.add_argument('--test-dir', default='', help='Optional explicit test directory override.')
    parser.add_argument('--class-name', default='', help='Optional true-label class filter inside the test split.')
    parser.add_argument('--sample-index', type=int, default=0, help='Sample index inside the filtered test set.')
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--list-classes', action='store_true')
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument('--image', default='', help='Optional explicit image path.')
    image_group.add_argument('--relative-path', default='', help='Optional relative path under test dir, e.g. glioma/xxx.jpg')
    return parser.parse_args()


def resolve_checkpoint_path(args: argparse.Namespace, script_stem: str) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    run_tag = sanitize_run_tag(args.run_tag)
    suffix = f'_{run_tag}' if run_tag else ''
    return (CHECKPOINT_DIR / f'best_{script_stem}_{args.loss}{suffix}.pt').resolve()


def main() -> None:
    args = parse_args()
    if args.list_models:
        print('Available Brain MRI model scripts:')
        for name in discover_model_scripts():
            print(f'  {name}')
        return

    ensure_pytorch_grad_cam()
    set_seed(SEED)
    test_dir = resolve_test_dir(args)
    records = build_test_records(test_dir)

    if args.list_classes:
        print('Available Brain MRI test classes:')
        for name in records['class_names']:
            print(f'  {name}')
        return

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_source = resolve_image_source(args, test_dir=test_dir, records=records)
    image_path = Path(str(image_source['image_path']))

    script_path = (THIS_DIR / args.model).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f'Model script not found: {script_path}')
    module = load_script_module(script_path, prefix='brain_mri_gradcam_pgc')
    build_model = getattr(module, 'build_model', None)
    if build_model is None:
        raise AttributeError(f'build_model() not found in {script_path.name}')

    checkpoint_path = resolve_checkpoint_path(args, script_path.stem)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    model = build_model(len(records['class_names'])).to(device)
    load_checkpoint_states(checkpoint_path, model, device, criterion=None)
    model.eval()

    target_module = resolve_target_module(
        model,
        args.target_layer,
        hint='Try layer4, layer3, layer2, inserted_module, or inserted_module.post_conv.',
    )

    transform = build_eval_transform(args.image_size)
    pil_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    pred_idx, pred_conf = predict(model, input_tensor, logits_extractor=extract_logits)

    if args.target_class is not None:
        target_class = int(args.target_class)
        cam_target_source = 'explicit'
    elif args.cam_on == 'true':
        if image_source['true_label_index'] is None:
            raise ValueError('cam-on=true requires a test-set image with a known true label.')
        target_class = int(image_source['true_label_index'])
        cam_target_source = 'true'
    else:
        target_class = pred_idx
        cam_target_source = 'pred'

    if target_class < 0 or target_class >= len(records['class_names']):
        raise ValueError(f'target-class out of range: {target_class}')

    original, heatmap, overlay, _ = build_cam_images(
        method_name=args.cam_method,
        model=model,
        target_module=target_module,
        input_tensor=input_tensor,
        class_idx=target_class,
        image_size=args.image_size,
        alpha=args.alpha,
        original_from_tensor=lambda t: tensor_to_pil(t, mean=MEAN, std=STD),
        aug_smooth=args.aug_smooth,
        eigen_smooth=args.eigen_smooth,
    )

    pred_label_name = records['class_names'][pred_idx]
    target_label_name = records['class_names'][target_class]
    info_lines = [
        f'model={script_path.stem} | checkpoint={checkpoint_path.name}',
        f'image={image_path.name} | target_layer={args.target_layer}',
        f'pred={pred_label_name} ({pred_idx}) | confidence={pred_conf:.4f}',
        f'cam_target={target_label_name} ({target_class}) | cam_on={cam_target_source} | cam_method={args.cam_method}',
        f"true={image_source['true_label_name']} ({image_source['true_label_index']})" if image_source['true_label_name'] is not None else 'true=unknown (custom image)',
    ]
    if args.aug_smooth or args.eigen_smooth:
        info_lines.append(f'aug_smooth={args.aug_smooth} | eigen_smooth={args.eigen_smooth}')
    panel = compose_panel(original, heatmap, overlay, info_lines)

    run_tag = sanitize_run_tag(args.run_tag)
    parts = [script_path.stem, args.loss]
    if run_tag:
        parts.append(run_tag)
    parts.extend([str(image_source['display_name']), args.target_layer.replace('.', '-'), args.cam_method.replace('+', 'plus'), f'camon-{cam_target_source}'])
    base_name = sanitize_filename('_'.join(parts))

    overlay_path = output_dir / f'{base_name}_overlay.png'
    heatmap_path = output_dir / f'{base_name}_heatmap.png'
    panel_path = output_dir / f'{base_name}_panel.png'
    overlay.save(overlay_path)
    heatmap.save(heatmap_path)
    panel.save(panel_path)

    print(f'Device: {device}')
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f'CUDA: {torch.cuda.get_device_name(0)}')
    print(f'Test dir: {test_dir}')
    print(f'Model script: {script_path.name}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Image: {image_path}')
    print(f'Target layer: {args.target_layer} -> {target_module.__class__.__name__}')
    print(f'CAM method: {args.cam_method} | cam_on={cam_target_source} | alpha={args.alpha:.2f}')
    if args.aug_smooth or args.eigen_smooth:
        print(f'CAM smoothing: aug_smooth={args.aug_smooth}, eigen_smooth={args.eigen_smooth}')
    print(f'Prediction: {pred_label_name} ({pred_idx}) | confidence={pred_conf:.4f}')
    if image_source['true_label_name'] is not None:
        print(f"True label: {image_source['true_label_name']} ({image_source['true_label_index']})")
    print(f'Target class for CAM: {target_label_name} ({target_class})')
    print(f'Saved overlay: {overlay_path}')
    print(f'Saved heatmap: {heatmap_path}')
    print(f'Saved panel: {panel_path}')


if __name__ == '__main__':
    main()
