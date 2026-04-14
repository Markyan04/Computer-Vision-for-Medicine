#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export Knee OA test samples where baseline fails but proposed model succeeds, with Grad-CAM visualizations."""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    GradCAM,
    build_eval_transform,
    build_test_records,
    find_last_conv,
    resolve_device,
    resolve_target_root,
    resolve_test_dir,
    tensor_to_pil,
)
from evaluate_test_checkpoint import (  # noqa: E402
    DEFAULT_CHECKPOINTS,
    build_model_from_module,
    load_knee_checkpoint_states,
    load_script_module,
    set_seed,
)
from gradcam_shared import build_heatmap_images, extract_logits, sanitize_filename  # noqa: E402


OUTPUT_ROOT = THIS_DIR / 'gradcam_comparison_exports'
DEFAULT_CLASS_NAMES = ['0_Normal', '1_Doubtful', '2_Mild', '3_Moderate', '4_Severe']


def discover_model_scripts() -> List[str]:
    return [path.name for path in sorted(THIS_DIR.glob('ResNet*.py'))]


def parse_args() -> argparse.Namespace:
    script_choices = discover_model_scripts()
    parser = argparse.ArgumentParser(
        description='Export test-set Grad-CAM comparison samples for Knee OA.',
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
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Inference device. Default: auto.',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=int(__import__('os').getenv('KNEE_IMAGE_SIZE', '224')),
        help='Resize size. Default reads KNEE_IMAGE_SIZE or falls back to 224.',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.45,
        help='Overlay alpha for the heatmap. Default: 0.45.',
    )
    parser.add_argument(
        '--data-root',
        default='',
        help='Optional Knee data root override. Defaults to KNEE_DATA_ROOT or PROJECT_ROOT/Knee_Osteoarthritis.',
    )
    parser.add_argument(
        '--test-dir',
        default='',
        help='Optional explicit test directory override.',
    )
    parser.add_argument(
        '--output-dir',
        default='',
        help='Optional output directory. Defaults to a timestamped folder under gradcam_comparison_exports/.',
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available model scripts and exit.',
    )
    parser.add_argument(
        '--list-classes',
        action='store_true',
        help='List available Knee OA classes and exit.',
    )

    parser.add_argument(
        '--baseline-model',
        default='ResNet_baseline.py',
        choices=script_choices,
        help='Baseline model script. Default: ResNet_baseline.py',
    )
    parser.add_argument(
        '--baseline-checkpoint',
        default='',
        help='Optional explicit baseline checkpoint path.',
    )
    parser.add_argument(
        '--baseline-target-layer',
        default='layer4',
        help='Baseline Grad-CAM target layer. Default: layer4.',
    )

    parser.add_argument(
        '--proposed-model',
        default='ResNet_layer3+MECS+Loss4.py',
        choices=script_choices,
        help='Proposed model script. Default: ResNet_layer3+MECS+Loss4.py',
    )
    parser.add_argument(
        '--proposed-checkpoint',
        default='',
        help='Optional explicit proposed checkpoint path.',
    )
    parser.add_argument(
        '--proposed-target-layer',
        default='layer4',
        help='Proposed Grad-CAM target layer. Default: layer4.',
    )
    return parser.parse_args()


def resolve_output_dir(raw: str, class_name: str) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_class = sanitize_filename(class_name)
    return (OUTPUT_ROOT / f'{safe_class}_{timestamp}').resolve()


def resolve_checkpoint(explicit_path: str, script_name: str) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    default_name = DEFAULT_CHECKPOINTS.get(script_name)
    if not default_name:
        raise ValueError(f'No default checkpoint mapping for {script_name}. Please pass an explicit checkpoint path.')
    return (THIS_DIR / default_name).resolve()


def load_model(script_name: str, checkpoint_path: Path, num_classes: int, device: torch.device):
    script_path = (THIS_DIR / script_name).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f'Model script not found: {script_path}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    module = load_script_module(script_path, prefix='knee_gradcam_compare')
    model = build_model_from_module(module, num_classes=num_classes).to(device)
    load_knee_checkpoint_states(checkpoint_path, model, device)
    model.eval()
    return module, model


def resolve_target_module(model: torch.nn.Module, target_layer: str) -> torch.nn.Module:
    target_root = resolve_target_root(model, target_layer)
    target_module = find_last_conv(target_root)
    if target_module is None:
        raise RuntimeError(
            f'No Conv2d layer found under target-layer={target_layer}. '
            'Try layer4, layer3, layer2, mecs, gcsa, or mdfa.'
        )
    return target_module


def predict(model: torch.nn.Module, input_tensor: torch.Tensor) -> Tuple[int, float]:
    with torch.no_grad():
        logits = extract_logits(model(input_tensor))
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        pred_conf = float(probs[0, pred_idx].item())
    return pred_idx, pred_conf


def generate_overlay(
    model: torch.nn.Module,
    target_module: torch.nn.Module,
    input_tensor: torch.Tensor,
    image_size: int,
    alpha: float,
    class_idx: int,
) -> Image.Image:
    gradcam = GradCAM(model, target_module)
    try:
        with torch.enable_grad():
            cam, _ = gradcam.generate(input_tensor, class_idx=class_idx)
    finally:
        gradcam.remove()

    original = tensor_to_pil(input_tensor.squeeze(0), mean=MEAN, std=STD).resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR,
    )
    _, overlay = build_heatmap_images(cam, original, alpha=alpha)
    return overlay


def compose_comparison_panel(
    original: Image.Image,
    baseline_overlay: Image.Image,
    proposed_overlay: Image.Image,
    info_lines: Sequence[str],
    baseline_title: str,
    proposed_title: str,
) -> Image.Image:
    margin = 20
    header_height = 120
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
        'baseline_prediction',
        'baseline_confidence',
        'proposed_prediction',
        'proposed_confidence',
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

    set_seed(1234)
    test_dir = resolve_test_dir(args)

    baseline_script_path = (THIS_DIR / args.baseline_model).resolve()
    baseline_module = load_script_module(baseline_script_path, prefix='knee_gradcam_classes')
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

    baseline_checkpoint = resolve_checkpoint(args.baseline_checkpoint, args.baseline_model)
    proposed_checkpoint = resolve_checkpoint(args.proposed_checkpoint, args.proposed_model)

    _, baseline_model = load_model(args.baseline_model, baseline_checkpoint, len(class_names), device)
    _, proposed_model = load_model(args.proposed_model, proposed_checkpoint, len(class_names), device)
    baseline_target_module = resolve_target_module(baseline_model, args.baseline_target_layer)
    proposed_target_module = resolve_target_module(proposed_model, args.proposed_target_layer)

    transform = build_eval_transform(args.image_size)

    print(f'Device: {device}')
    print(f'Resolved test dir: {test_dir}')
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f'CUDA: {torch.cuda.get_device_name(0)}')
    print(f'Test split size: {len(records["rows"])}')
    print(f'Candidate class: {args.class_name} | candidates in test split: {len(candidate_rows)}')
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

        original = tensor_to_pil(input_tensor.squeeze(0), mean=MEAN, std=STD).resize(
            (args.image_size, args.image_size),
            resample=Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR,
        )
        baseline_overlay = generate_overlay(
            baseline_model,
            baseline_target_module,
            input_tensor,
            image_size=args.image_size,
            alpha=args.alpha,
            class_idx=baseline_pred_idx,
        )
        proposed_overlay = generate_overlay(
            proposed_model,
            proposed_target_module,
            input_tensor,
            image_size=args.image_size,
            alpha=args.alpha,
            class_idx=proposed_pred_idx,
        )

        baseline_pred_name = class_names[baseline_pred_idx]
        proposed_pred_name = class_names[proposed_pred_idx]
        true_label_name = class_names[true_idx]
        true_folder_name = str(row['folder_name'])
        image_id = Path(str(row['relative_path'])).stem

        info_lines = [
            f'image_id={image_id} | true={true_label_name} ({true_folder_name})',
            f'baseline={baseline_pred_name} ({baseline_conf:.4f}) | proposed={proposed_pred_name} ({proposed_conf:.4f})',
            f'baseline_layer={args.baseline_target_layer} | proposed_layer={args.proposed_target_layer}',
        ]
        panel = compose_comparison_panel(
            original,
            baseline_overlay,
            proposed_overlay,
            info_lines=info_lines,
            baseline_title=f'Baseline ({baseline_pred_name})',
            proposed_title=f'Proposed ({proposed_pred_name})',
        )

        stem = sanitize_filename(
            f'{image_id}_true-{true_label_name}_base-{baseline_pred_name}_prop-{proposed_pred_name}'
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
            'baseline_prediction': baseline_pred_name,
            'baseline_confidence': f'{baseline_conf:.6f}',
            'proposed_prediction': proposed_pred_name,
            'proposed_confidence': f'{proposed_conf:.6f}',
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
