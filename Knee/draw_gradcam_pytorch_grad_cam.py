#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Draw Knee OA Grad-CAM visualizations with pytorch-grad-cam."""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from PIL import Image
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from draw_gradcam import (  # noqa: E402
    MEAN,
    STD,
    build_eval_transform,
    build_test_records,
    compose_panel,
    discover_model_scripts,
    resolve_device,
    resolve_image_source,
    resolve_test_dir,
)
from evaluate_test_checkpoint import (  # noqa: E402
    DEFAULT_CHECKPOINTS,
    build_model_from_module,
    load_knee_checkpoint_states,
    load_script_module,
    set_seed,
)
from gradcam_shared import sanitize_filename  # noqa: E402
from pytorch_grad_cam_utils import (  # noqa: E402
    CAM_METHOD_CHOICES,
    build_cam_images,
    ensure_pytorch_grad_cam,
    predict,
    resolve_checkpoint_path,
    resolve_target_module,
)


OUTPUT_DIR = THIS_DIR / 'gradcam_outputs_pytorch_grad_cam'


def parse_args() -> argparse.Namespace:
    script_choices = discover_model_scripts()
    parser = argparse.ArgumentParser(description='Draw Knee OA Grad-CAM with pytorch-grad-cam.')
    parser.add_argument('--model', default='ResNet_layer3+MECS+Loss4.py', choices=script_choices)
    parser.add_argument('--checkpoint', default='', help='Optional explicit checkpoint path.')
    parser.add_argument('--target-layer', default='layer4', help='Layer/module path to visualize, e.g. layer4, layer3, mecs, gcsa, or mdfa.')
    parser.add_argument('--target-class', type=int, default=None, help='Optional target class index. Overrides --cam-on.')
    parser.add_argument('--cam-on', default='pred', choices=['pred', 'true'], help='Which class to explain when --target-class is not provided.')
    parser.add_argument('--cam-method', default='gradcam++', choices=CAM_METHOD_CHOICES, help='pytorch-grad-cam method. Default: gradcam++.')
    parser.add_argument('--aug-smooth', action='store_true', help='Enable test-time augmentation smoothing if supported.')
    parser.add_argument('--eigen-smooth', action='store_true', help='Enable eigen smoothing if supported.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--image-size', type=int, default=int(os.getenv('KNEE_IMAGE_SIZE', '224')))
    parser.add_argument('--alpha', type=float, default=0.55, help='Heatmap opacity. Default: 0.35.')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    parser.add_argument('--data-root', default='', help='Optional Knee data root override.')
    parser.add_argument('--test-dir', default='', help='Optional explicit test directory override.')
    parser.add_argument('--class-name', default='', help='Optional true-label class filter. Supports folder names and display names.')
    parser.add_argument('--sample-index', type=int, default=0, help='Sample index inside the filtered test set.')
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--list-classes', action='store_true')
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument('--image', default='', help='Optional explicit image path.')
    image_group.add_argument('--relative-path', default='', help='Optional relative path under test dir, e.g. 2/xxx.png')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_models:
        print('Available Knee model scripts:')
        for name in discover_model_scripts():
            print(f'  {name}')
        return

    ensure_pytorch_grad_cam()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = (THIS_DIR / args.model).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f'Model script not found: {script_path}')
    module = load_script_module(script_path, prefix='knee_gradcam_pgc')
    set_seed(int(getattr(module, 'SEED', 1234)))

    class_names = list(getattr(module, 'CLASS_NAMES', ['0_Normal', '1_Doubtful', '2_Mild', '3_Moderate', '4_Severe']))
    test_dir = resolve_test_dir(args)
    records = build_test_records(test_dir, class_names=class_names)

    if args.list_classes:
        print('Available Knee test classes:')
        for index, name in enumerate(class_names):
            folder_name = records['dataset'].classes[index] if index < len(records['dataset'].classes) else str(index)
            print(f'  {folder_name} -> {name}')
        return

    image_source = resolve_image_source(args, test_dir=test_dir, records=records)
    image_path = Path(str(image_source['image_path']))
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.model, DEFAULT_CHECKPOINTS, THIS_DIR)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    model = build_model_from_module(module, num_classes=len(class_names)).to(device)
    load_knee_checkpoint_states(checkpoint_path, model, device)
    model.eval()

    target_module = resolve_target_module(model, args.target_layer)
    transform = build_eval_transform(args.image_size)
    pil_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    pred_idx, pred_conf = predict(model, input_tensor)

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

    if target_class < 0 or target_class >= len(class_names):
        raise ValueError(f'target-class out of range: {target_class}')

    original, heatmap, overlay, _ = build_cam_images(
        method_name=args.cam_method,
        model=model,
        target_module=target_module,
        input_tensor=input_tensor,
        class_idx=target_class,
        image_size=args.image_size,
        alpha=args.alpha,
        mean=MEAN,
        std=STD,
        aug_smooth=args.aug_smooth,
        eigen_smooth=args.eigen_smooth,
    )

    pred_label_name = class_names[pred_idx]
    target_label_name = class_names[target_class]
    info_lines = [
        f'model={script_path.stem} | checkpoint={checkpoint_path.name}',
        f'image={image_path.name} | target_layer={args.target_layer} | cam_method={args.cam_method}',
        f'pred={pred_label_name} ({pred_idx}) | confidence={pred_conf:.4f}',
        f'cam_target={target_label_name} ({target_class}) | cam_on={cam_target_source}',
        f"true={image_source['true_label_name']} ({image_source['true_label_index']})" if image_source['true_label_name'] is not None else 'true=unknown (custom image)',
    ]
    if args.aug_smooth or args.eigen_smooth:
        info_lines.append(f'aug_smooth={args.aug_smooth} | eigen_smooth={args.eigen_smooth}')
    panel = compose_panel(original, heatmap, overlay, info_lines)

    base_name = sanitize_filename(
        '_'.join([
            script_path.stem,
            image_source['display_name'],
            args.target_layer.replace('.', '-'),
            args.cam_method.replace('+', 'plus'),
            f'camon-{cam_target_source}',
        ])
    )
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
