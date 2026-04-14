#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export Grad-CAM visualizations for every Knee OA test image of a chosen class."""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


OUTPUT_ROOT = THIS_DIR / 'gradcam_class_exports_pytorch_grad_cam'
DEFAULT_CLASS_NAMES = ['0_Normal', '1_Doubtful', '2_Mild', '3_Moderate', '4_Severe']


def parse_args() -> argparse.Namespace:
    script_choices = discover_model_scripts()
    parser = argparse.ArgumentParser(
        description='Export Grad-CAM visualizations for all Knee OA test images of a chosen class.'
    )
    parser.add_argument('--class-name', required=True, help='True-label class filter. Supports folder names (0-4) and display names such as 2_Mild.')
    parser.add_argument('--model', default='ResNet_layer3+MECS+Loss4.py', choices=script_choices)
    parser.add_argument('--checkpoint', default='', help='Optional explicit checkpoint path.')
    parser.add_argument('--target-layer', default='layer4', help='Layer/module path to visualize, e.g. layer4, layer3, mecs, gcsa, or mdfa.')
    parser.add_argument('--target-class', type=int, default=None, help='Optional target class index. Overrides --cam-on.')
    parser.add_argument('--cam-on', default='true', choices=['pred', 'true'], help='Which class to explain when --target-class is not provided. Default: true.')
    parser.add_argument('--cam-method', default='gradcam++', choices=CAM_METHOD_CHOICES, help='pytorch-grad-cam method. Default: gradcam++.')
    parser.add_argument('--aug-smooth', action='store_true', help='Enable test-time augmentation smoothing if supported.')
    parser.add_argument('--eigen-smooth', action='store_true', help='Enable eigen smoothing if supported.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--image-size', type=int, default=int(os.getenv('KNEE_IMAGE_SIZE', '224')))
    parser.add_argument('--alpha', type=float, default=0.35, help='Heatmap opacity. Default: 0.35.')
    parser.add_argument('--data-root', default='', help='Optional Knee data root override.')
    parser.add_argument('--test-dir', default='', help='Optional explicit test directory override.')
    parser.add_argument('--output-dir', default='', help='Optional output directory. Defaults to a timestamped folder under gradcam_class_exports_pytorch_grad_cam/.')
    parser.add_argument('--max-samples', type=int, default=0, help='Maximum number of images to export. Use <=0 for all. Default: 0.')
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--list-classes', action='store_true')
    return parser.parse_args()


def resolve_output_dir(raw: str, class_name: str, model_stem: str) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_class = sanitize_filename(class_name)
    safe_model = sanitize_filename(model_stem)
    return (OUTPUT_ROOT / f'{safe_model}_{safe_class}_{timestamp}').resolve()


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
        'image_id', 'true_label', 'true_folder', 'image_path', 'cam_method', 'cam_on',
        'prediction', 'prediction_confidence', 'cam_target',
        'original_path', 'heatmap_path', 'overlay_path', 'panel_path',
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
    device = resolve_device(args.device)

    script_path = (THIS_DIR / args.model).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f'Model script not found: {script_path}')
    module = load_script_module(script_path, prefix='knee_gradcam_class_pgc')
    set_seed(int(getattr(module, 'SEED', 1234)))

    class_names = list(getattr(module, 'CLASS_NAMES', DEFAULT_CLASS_NAMES))
    test_dir = resolve_test_dir(args)
    records = build_test_records(test_dir, class_names=class_names)

    if args.list_classes:
        print('Available Knee test classes:')
        for index, name in enumerate(class_names):
            folder_name = records['dataset'].classes[index] if index < len(records['dataset'].classes) else str(index)
            print(f'  {folder_name} -> {name}')
        return

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.model, DEFAULT_CHECKPOINTS, THIS_DIR)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    output_dir = resolve_output_dir(args.output_dir, args.class_name, script_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_module(module, num_classes=len(class_names)).to(device)
    load_knee_checkpoint_states(checkpoint_path, model, device)
    model.eval()

    target_module = resolve_target_module(model, args.target_layer)
    transform = build_eval_transform(args.image_size)
    candidate_rows = collect_class_records(records, args.class_name)

    print(f'Device: {device}')
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f'CUDA: {torch.cuda.get_device_name(0)}')
    print(f'Test dir: {test_dir}')
    print(f'Model script: {script_path.name}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Test split size: {len(records["rows"])}')
    print(f'Candidate class: {args.class_name} | candidates in test split: {len(candidate_rows)}')
    print(f'CAM method: {args.cam_method} | cam_on={args.cam_on} | alpha={args.alpha:.2f}')
    if args.aug_smooth or args.eigen_smooth:
        print(f'CAM smoothing: aug_smooth={args.aug_smooth}, eigen_smooth={args.eigen_smooth}')
    print(f'Target layer: {args.target_layer} -> {target_module.__class__.__name__}')
    print(f'Output dir: {output_dir}')

    if not candidate_rows:
        print('No matching test samples found. Nothing was exported.')
        return

    summary_rows: List[Dict[str, object]] = []
    exported = 0

    for row in candidate_rows:
        image_path = Path(str(row['image_path']))
        pil_image = Image.open(image_path).convert('RGB')
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        pred_idx, pred_conf = predict(model, input_tensor)
        true_idx = int(row['label_index'])

        if args.target_class is not None:
            cam_target_idx = int(args.target_class)
            cam_target_source = 'explicit'
        elif args.cam_on == 'true':
            cam_target_idx = true_idx
            cam_target_source = 'true'
        else:
            cam_target_idx = pred_idx
            cam_target_source = 'pred'

        if cam_target_idx < 0 or cam_target_idx >= len(class_names):
            raise ValueError(f'target-class out of range: {cam_target_idx}')

        original, heatmap, overlay, _ = build_cam_images(
            method_name=args.cam_method,
            model=model,
            target_module=target_module,
            input_tensor=input_tensor,
            class_idx=cam_target_idx,
            image_size=args.image_size,
            alpha=args.alpha,
            mean=MEAN,
            std=STD,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )

        pred_label_name = class_names[pred_idx]
        true_label_name = class_names[true_idx]
        cam_target_name = class_names[cam_target_idx]
        true_folder_name = str(row['folder_name'])
        image_id = Path(str(row['relative_path'])).stem

        info_lines = [
            f'image_id={image_id} | true={true_label_name} ({true_folder_name})',
            f'pred={pred_label_name} ({pred_conf:.4f}) | cam_target={cam_target_name}',
            f'cam_method={args.cam_method} | cam_on={cam_target_source} | layer={args.target_layer}',
        ]
        if args.aug_smooth or args.eigen_smooth:
            info_lines.append(f'aug_smooth={args.aug_smooth} | eigen_smooth={args.eigen_smooth}')
        panel = compose_panel(original, heatmap, overlay, info_lines)

        stem = sanitize_filename(
            f'{image_id}_true-{true_label_name}_pred-{pred_label_name}_{args.target_layer.replace(".", "-")}_{args.cam_method.replace("+", "plus")}_camon-{cam_target_source}'
        )
        original_path = output_dir / f'{stem}_original.png'
        heatmap_path = output_dir / f'{stem}_heatmap.png'
        overlay_path = output_dir / f'{stem}_overlay.png'
        panel_path = output_dir / f'{stem}_panel.png'

        original.save(original_path)
        heatmap.save(heatmap_path)
        overlay.save(overlay_path)
        panel.save(panel_path)

        summary_rows.append({
            'image_id': image_id,
            'true_label': true_label_name,
            'true_folder': true_folder_name,
            'image_path': str(image_path),
            'cam_method': args.cam_method,
            'cam_on': cam_target_source,
            'prediction': pred_label_name,
            'prediction_confidence': f'{pred_conf:.6f}',
            'cam_target': cam_target_name,
            'original_path': str(original_path),
            'heatmap_path': str(heatmap_path),
            'overlay_path': str(overlay_path),
            'panel_path': str(panel_path),
        })
        exported += 1
        print(f'[{exported}] exported image_id={image_id} | true={true_label_name} | pred={pred_label_name}')

        if args.max_samples > 0 and exported >= args.max_samples:
            break

    summary_path = write_summary(summary_rows, output_dir)
    print(f'Exported {len(summary_rows)} sample(s) to: {output_dir}')
    if summary_path is not None:
        print(f'Summary CSV: {summary_path}')


if __name__ == '__main__':
    main()
