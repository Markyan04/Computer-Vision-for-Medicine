#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export Brain Tumor MRI test samples where baseline fails but proposed model succeeds, with Grad-CAM visualizations."""

import argparse
import csv
import os
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

from brain_tumor_mri_loss_experiment_common import (  # noqa: E402
    SEED,
    load_checkpoint_states,
    sanitize_run_tag,
    set_seed,
)
from draw_gradcam import (  # noqa: E402
    CHECKPOINT_DIR,
    MEAN,
    STD,
    GradCAM,
    build_eval_transform,
    build_heatmap_images,
    build_test_records,
    extract_logits,
    find_last_conv,
    load_script_module,
    resolve_device,
    resolve_target_root,
    resolve_test_dir,
    sanitize_filename,
    tensor_to_pil,
)


OUTPUT_ROOT = THIS_DIR / 'gradcam_comparison_exports'


def discover_model_scripts() -> List[str]:
    return [path.name for path in sorted(THIS_DIR.glob('ResNet*.py'))]


def parse_args() -> argparse.Namespace:
    script_choices = discover_model_scripts()
    parser = argparse.ArgumentParser(
        description='Export test-set Grad-CAM comparison samples for Brain Tumor MRI.',
    )
    parser.add_argument(
        '--class-name',
        help='True-label class name to filter, e.g. glioma, meningioma, notumor, pituitary.',
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
        default=int(os.getenv('BRAIN_MRI_IMAGE_SIZE', '224')),
        help='Resize size. Default reads BRAIN_MRI_IMAGE_SIZE or falls back to 224.',
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
        help='Optional Brain MRI data root override. Defaults to BRAIN_MRI_DATA_ROOT or PROJECT_ROOT/Brain_Tumor_MRI.',
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
        help='List available Brain Tumor MRI classes and exit.',
    )

    parser.add_argument(
        '--baseline-model',
        default='ResNet_baseline.py',
        choices=script_choices,
        help='Baseline model script. Default: ResNet_baseline.py',
    )
    parser.add_argument(
        '--baseline-loss',
        default='ce',
        help='Baseline checkpoint loss suffix. Default: ce.',
    )
    parser.add_argument(
        '--baseline-run-tag',
        default='',
        help='Optional baseline run tag used in checkpoint naming.',
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
        default='ResNet_layer3+MECS.py',
        choices=script_choices,
        help='Proposed model script. Default: ResNet_layer3+MECS.py',
    )
    parser.add_argument(
        '--proposed-loss',
        default='dast',
        help='Proposed checkpoint loss suffix. Default: dast.',
    )
    parser.add_argument(
        '--proposed-run-tag',
        default='',
        help='Optional proposed run tag used in checkpoint naming.',
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


def resolve_checkpoint(explicit_path: str, script_name: str, loss_name: str, run_tag: str) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    script_stem = Path(script_name).stem
    cleaned_tag = sanitize_run_tag(run_tag)
    suffix = f'_{cleaned_tag}' if cleaned_tag else ''
    return CHECKPOINT_DIR / f'best_{script_stem}_{loss_name}{suffix}.pt'


def load_model(script_name: str, checkpoint_path: Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    script_path = (THIS_DIR / script_name).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f'Model script not found: {script_path}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    module = load_script_module(script_path, prefix='brain_mri_gradcam_compare')
    build_model = getattr(module, 'build_model', None)
    if build_model is None:
        raise AttributeError(f'build_model() not found in: {script_path}')

    model = build_model(num_classes).to(device)
    load_checkpoint_states(checkpoint_path, model, device, criterion=None)
    model.eval()
    return model


def resolve_target_module(model: torch.nn.Module, target_layer: str) -> torch.nn.Module:
    target_root = resolve_target_root(model, target_layer)
    target_module = find_last_conv(target_root)
    if target_module is None:
        raise RuntimeError(
            f'No Conv2d layer found under target-layer={target_layer}. '
            'Try layer4, layer3, layer2, inserted_module, or inserted_module.post_conv.'
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
        if str(row['label_name']).strip().lower() == target:
            rows.append(row)
    return rows


def write_summary(summary_rows: Sequence[Dict[str, object]], output_dir: Path) -> Optional[Path]:
    if not summary_rows:
        return None

    preferred = [
        'image_id',
        'true_label',
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
        print('Available Brain Tumor MRI model scripts:')
        for name in discover_model_scripts():
            print(f'  {name}')
        return

    set_seed(SEED)
    test_dir = resolve_test_dir(args)
    records = build_test_records(test_dir)
    class_names = list(records['class_names'])

    if args.list_classes:
        print('Available Brain Tumor MRI classes:')
        for name in class_names:
            print(f'  {name}')
        return

    if not args.class_name:
        print('Missing required argument: --class-name')
        print('Use --list-classes to inspect available Brain Tumor MRI labels.')
        return

    class_lookup = {name.lower(): idx for idx, name in enumerate(class_names)}
    target_class_name = args.class_name.strip().lower()
    if target_class_name not in class_lookup:
        print(f'Resolved test dir: {test_dir}')
        print(f'Unknown class-name: {args.class_name}')
        print('Available classes: ' + ', '.join(class_names))
        return

    device = resolve_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, target_class_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_checkpoint = resolve_checkpoint(
        explicit_path=args.baseline_checkpoint,
        script_name=args.baseline_model,
        loss_name=args.baseline_loss,
        run_tag=args.baseline_run_tag,
    )
    proposed_checkpoint = resolve_checkpoint(
        explicit_path=args.proposed_checkpoint,
        script_name=args.proposed_model,
        loss_name=args.proposed_loss,
        run_tag=args.proposed_run_tag,
    )

    baseline_model = load_model(args.baseline_model, baseline_checkpoint, len(class_names), device)
    proposed_model = load_model(args.proposed_model, proposed_checkpoint, len(class_names), device)
    baseline_target_module = resolve_target_module(baseline_model, args.baseline_target_layer)
    proposed_target_module = resolve_target_module(proposed_model, args.proposed_target_layer)

    transform = build_eval_transform(args.image_size)
    candidate_rows = collect_class_records(records, target_class_name)

    print(f'Device: {device}')
    print(f'Resolved test dir: {test_dir}')
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f'CUDA: {torch.cuda.get_device_name(0)}')
    print(f'Test split size: {len(records["rows"])}')
    print(f'Candidate class: {target_class_name} | candidates in test split: {len(candidate_rows)}')
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
        image_id = Path(str(row['relative_path'])).stem

        info_lines = [
            f'image_id={image_id} | true={true_label_name}',
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
            f'No matching test samples found for class={target_class_name}. '
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

