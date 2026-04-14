#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export Knee OA test samples where baseline fails but proposed succeeds, using pytorch-grad-cam."""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageDraw
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
    discover_model_scripts,
    resolve_device,
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


OUTPUT_ROOT = THIS_DIR / 'gradcam_comparison_exports_pytorch_grad_cam'
DEFAULT_CLASS_NAMES = ['0_Normal', '1_Doubtful', '2_Mild', '3_Moderate', '4_Severe']


def parse_args() -> argparse.Namespace:
    script_choices = discover_model_scripts()
    parser = argparse.ArgumentParser(
        description='Export Knee OA test-set Grad-CAM comparison samples with pytorch-grad-cam.',
    )
    parser.add_argument(
        '--class-name',
        help='True-label class filter. Supports both folder names (0-4) and display names such as 2_Mild.',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10,
        help='Maximum number of matching samples to export. Use <=0 for all. Default: 10.',
    )
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--image-size', type=int, default=int(os.getenv('KNEE_IMAGE_SIZE', '224')))
    parser.add_argument('--alpha', type=float, default=0.35, help='Heatmap opacity. Default: 0.35.')
    parser.add_argument('--cam-method', default='gradcam++', choices=CAM_METHOD_CHOICES, help='pytorch-grad-cam method. Default: gradcam++.')
    parser.add_argument('--cam-on', default='pred', choices=['pred', 'true'], help='Draw CAM for each model\'s prediction or for the shared true label.')
    parser.add_argument('--aug-smooth', action='store_true', help='Enable test-time augmentation smoothing if supported.')
    parser.add_argument('--eigen-smooth', action='store_true', help='Enable eigen smoothing if supported.')
    parser.add_argument('--data-root', default='', help='Optional Knee data root override.')
    parser.add_argument('--test-dir', default='', help='Optional explicit test directory override.')
    parser.add_argument('--output-dir', default='', help='Optional output directory. Defaults to a timestamped folder under gradcam_comparison_exports_pytorch_grad_cam/.')
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--list-classes', action='store_true')

    parser.add_argument('--baseline-model', default='ResNet_baseline.py', choices=script_choices)
    parser.add_argument('--baseline-checkpoint', default='', help='Optional explicit baseline checkpoint path.')
    parser.add_argument('--baseline-target-layer', default='layer4', help='Baseline Grad-CAM target layer. Default: layer4.')

    parser.add_argument('--proposed-model', default='ResNet_layer3+MECS+Loss4.py', choices=script_choices)
    parser.add_argument('--proposed-checkpoint', default='', help='Optional explicit proposed checkpoint path.')
    parser.add_argument('--proposed-target-layer', default='layer4', help='Proposed Grad-CAM target layer. Default: layer4.')
    return parser.parse_args()


def resolve_output_dir(raw: str, class_name: str) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_class = sanitize_filename(class_name)
    return (OUTPUT_ROOT / f'{safe_class}_{timestamp}').resolve()


def compose_comparison_panel(
    original: Image.Image,
    baseline_overlay: Image.Image,
    proposed_overlay: Image.Image,
    info_lines: Sequence[str],
    baseline_title: str,
    proposed_title: str,
) -> Image.Image:
    margin = 20
    header_height = 130
    panel_width = original.width * 3 + margin * 4
    panel_height = header_height + original.height + margin

    canvas = Image.new('RGB', (panel_width, panel_height), color='white')
    draw = ImageDraw.Draw(canvas)

    y = 12
    for line in info_lines:
        draw.text((margin, y), line, fill='black')
        y += 18

    positions = [margin, margin * 2 + original.width, margin * 3 + original.width * 2]
    titles = ['Original', baseline_title, proposed_title]
    images = [original.convert('RGB'), baseline_overlay.convert('RGB'), proposed_overlay.convert('RGB')]

    for x, title, image in zip(positions, titles, images):
        draw.text((x, header_height - 24), title, fill='black')
        canvas.paste(image, (x, header_height))

    return canvas


def collect_class_records(records: Dict[str, object], class_name: str) -> List[Dict[str, object]]:
    target = class_name.strip().lower()
    rows = []
    for row in records['rows']:
        if str(row['label_name']).lower() == target or str(row['folder_name']).lower() == target:
            rows.append(row)
    return rows


def write_summary(summary_rows: Sequence[Dict[str, object]], output_dir: Path) -> Optional[Path]:
    if not summary_rows:
        return None

    preferred = [
        'image_id',
        'true_label',
        'true_folder',
        'image_path',
        'cam_method',
        'cam_on',
        'baseline_prediction',
        'baseline_confidence',
        'baseline_cam_target',
        'proposed_prediction',
        'proposed_confidence',
        'proposed_cam_target',
        'original_path',
        'baseline_gradcam_path',
        'proposed_gradcam_path',
        'panel_path',
    ]
    seen: List[str] = []
    for row in summary_rows:
        for key in row.keys():
            if key not in seen:
                seen.append(key)
    fieldnames = [field for field in preferred if field in seen]
    fieldnames.extend(field for field in seen if field not in fieldnames)

    summary_path = output_dir / 'summary.csv'
    with open(summary_path, 'w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return summary_path


def main() -> None:
    args = parse_args()
    if args.list_models:
        print('Available Knee model scripts:')
        for name in discover_model_scripts():
            print(f'  {name}')
        return

    ensure_pytorch_grad_cam()
    set_seed(1234)
    test_dir = resolve_test_dir(args)

    baseline_script_path = (THIS_DIR / args.baseline_model).resolve()
    baseline_module = load_script_module(baseline_script_path, prefix='knee_gradcam_pgc_classes')
    class_names = list(getattr(baseline_module, 'CLASS_NAMES', DEFAULT_CLASS_NAMES))
    records = build_test_records(test_dir, class_names=class_names)

    if args.list_classes:
        print('Available Knee classes:')
        for index, name in enumerate(class_names):
            folder_name = records['dataset'].classes[index] if index < len(records['dataset'].classes) else str(index)
            print(f'  {folder_name} -> {name}')
        return

    if not args.class_name:
        print('Missing required argument: --class-name')
        print('Use --list-classes to inspect available Knee labels.')
        return

    candidate_rows = collect_class_records(records, args.class_name)
    if not candidate_rows:
        available = [f"{folder} -> {label}" for folder, label in zip(records['dataset'].classes, class_names)]
        print(f'Resolved test dir: {test_dir}')
        print(f'Unknown class-name or no samples found: {args.class_name}')
        print('Available classes: ' + ', '.join(available))
        return

    device = resolve_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, args.class_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_checkpoint = resolve_checkpoint_path(args.baseline_checkpoint, args.baseline_model, DEFAULT_CHECKPOINTS, THIS_DIR)
    proposed_checkpoint = resolve_checkpoint_path(args.proposed_checkpoint, args.proposed_model, DEFAULT_CHECKPOINTS, THIS_DIR)

    baseline_model_module = load_script_module((THIS_DIR / args.baseline_model).resolve(), prefix='knee_gradcam_pgc_base')
    proposed_model_module = load_script_module((THIS_DIR / args.proposed_model).resolve(), prefix='knee_gradcam_pgc_prop')

    baseline_model = build_model_from_module(baseline_model_module, num_classes=len(class_names)).to(device)
    proposed_model = build_model_from_module(proposed_model_module, num_classes=len(class_names)).to(device)
    load_knee_checkpoint_states(baseline_checkpoint, baseline_model, device)
    load_knee_checkpoint_states(proposed_checkpoint, proposed_model, device)
    baseline_model.eval()
    proposed_model.eval()

    baseline_target_module = resolve_target_module(baseline_model, args.baseline_target_layer)
    proposed_target_module = resolve_target_module(proposed_model, args.proposed_target_layer)
    transform = build_eval_transform(args.image_size)

    print(f'Device: {device}')
    print(f'Resolved test dir: {test_dir}')
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f'CUDA: {torch.cuda.get_device_name(0)}')
    print(f'Test split size: {len(records["rows"])}')
    print(f'Candidate class: {args.class_name} | candidates in test split: {len(candidate_rows)}')
    print(f'CAM method: {args.cam_method} | cam_on={args.cam_on} | alpha={args.alpha:.2f}')
    if args.aug_smooth or args.eigen_smooth:
        print(f'CAM smoothing: aug_smooth={args.aug_smooth}, eigen_smooth={args.eigen_smooth}')
    print(f'Baseline : {args.baseline_model} | checkpoint={baseline_checkpoint.name} | target_layer={args.baseline_target_layer}')
    print(f'Proposed : {args.proposed_model} | checkpoint={proposed_checkpoint.name} | target_layer={args.proposed_target_layer}')
    print(f'Output dir: {output_dir}')

    max_samples = args.max_samples
    summary_rows: List[Dict[str, object]] = []
    exported = 0

    for row in candidate_rows:
        image_path = Path(str(row['image_path']))
        pil_image = Image.open(image_path).convert('RGB')
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        baseline_pred_idx, baseline_conf = predict(baseline_model, input_tensor)
        proposed_pred_idx, proposed_conf = predict(proposed_model, input_tensor)

        true_idx = int(row['label_index'])
        baseline_correct = baseline_pred_idx == true_idx
        proposed_correct = proposed_pred_idx == true_idx
        if baseline_correct or not proposed_correct:
            continue

        if args.cam_on == 'true':
            baseline_cam_idx = true_idx
            proposed_cam_idx = true_idx
        else:
            baseline_cam_idx = baseline_pred_idx
            proposed_cam_idx = proposed_pred_idx

        original, _, baseline_overlay, _ = build_cam_images(
            method_name=args.cam_method,
            model=baseline_model,
            target_module=baseline_target_module,
            input_tensor=input_tensor,
            class_idx=baseline_cam_idx,
            image_size=args.image_size,
            alpha=args.alpha,
            mean=MEAN,
            std=STD,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )
        _, _, proposed_overlay, _ = build_cam_images(
            method_name=args.cam_method,
            model=proposed_model,
            target_module=proposed_target_module,
            input_tensor=input_tensor,
            class_idx=proposed_cam_idx,
            image_size=args.image_size,
            alpha=args.alpha,
            mean=MEAN,
            std=STD,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )

        baseline_pred_name = class_names[baseline_pred_idx]
        proposed_pred_name = class_names[proposed_pred_idx]
        baseline_cam_name = class_names[baseline_cam_idx]
        proposed_cam_name = class_names[proposed_cam_idx]
        true_label_name = class_names[true_idx]
        true_folder_name = str(row['folder_name'])
        image_id = Path(str(row['relative_path'])).stem

        info_lines = [
            f'image_id={image_id} | true={true_label_name} ({true_folder_name})',
            f'baseline={baseline_pred_name} ({baseline_conf:.4f}) | proposed={proposed_pred_name} ({proposed_conf:.4f})',
            f'cam_method={args.cam_method} | cam_on={args.cam_on}',
            f'baseline_layer={args.baseline_target_layer} -> {baseline_cam_name} | proposed_layer={args.proposed_target_layer} -> {proposed_cam_name}',
        ]
        if args.aug_smooth or args.eigen_smooth:
            info_lines.append(f'aug_smooth={args.aug_smooth} | eigen_smooth={args.eigen_smooth}')
        panel = compose_comparison_panel(
            original,
            baseline_overlay,
            proposed_overlay,
            info_lines=info_lines,
            baseline_title=f'Baseline ({baseline_pred_name})',
            proposed_title=f'Proposed ({proposed_pred_name})',
        )

        stem = sanitize_filename(
            f'{image_id}_true-{true_label_name}_base-{baseline_pred_name}_prop-{proposed_pred_name}_{args.cam_method.replace("+", "plus")}_{args.cam_on}'
        )
        original_path = output_dir / f'{stem}_original.png'
        baseline_path = output_dir / f'{stem}_baseline_gradcam.png'
        proposed_path = output_dir / f'{stem}_proposed_gradcam.png'
        panel_path = output_dir / f'{stem}_panel.png'

        original.save(original_path)
        baseline_overlay.save(baseline_path)
        proposed_overlay.save(proposed_path)
        panel.save(panel_path)

        summary_rows.append({
            'image_id': image_id,
            'true_label': true_label_name,
            'true_folder': true_folder_name,
            'image_path': str(image_path),
            'cam_method': args.cam_method,
            'cam_on': args.cam_on,
            'baseline_prediction': baseline_pred_name,
            'baseline_confidence': f'{baseline_conf:.6f}',
            'baseline_cam_target': baseline_cam_name,
            'proposed_prediction': proposed_pred_name,
            'proposed_confidence': f'{proposed_conf:.6f}',
            'proposed_cam_target': proposed_cam_name,
            'original_path': str(original_path),
            'baseline_gradcam_path': str(baseline_path),
            'proposed_gradcam_path': str(proposed_path),
            'panel_path': str(panel_path),
        })
        exported += 1
        print(
            f'[{exported}] exported image_id={image_id} | '
            f'true={true_label_name} | baseline={baseline_pred_name} | proposed={proposed_pred_name}'
        )

        if max_samples > 0 and exported >= max_samples:
            break

    if not summary_rows:
        print(
            f'No matching test samples found for class={args.class_name}. '
            'Condition: baseline predicts wrong while proposed predicts correct.'
        )
        print('Nothing was exported.')
        return

    summary_path = write_summary(summary_rows, output_dir)
    print(f'Exported {len(summary_rows)} sample(s) to: {output_dir}')
    if summary_path is not None:
        print(f'Summary CSV: {summary_path}')


if __name__ == '__main__':
    main()
