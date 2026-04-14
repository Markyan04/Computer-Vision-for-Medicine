#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Draw Grad-CAM visualizations for Knee OA checkpoints."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from gradcam_shared import (  # noqa: E402
    BILINEAR,
    GradCAM,
    build_heatmap_images,
    compose_panel,
    extract_logits,
    find_last_conv,
    load_script_module,
    resolve_device,
    resolve_target_root,
    sanitize_filename,
    tensor_to_pil,
)


OUTPUT_DIR = THIS_DIR / 'gradcam_outputs'
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
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


def discover_model_scripts() -> List[str]:
    return [path.name for path in sorted(THIS_DIR.glob('ResNet*.py'))]


def parse_args() -> argparse.Namespace:
    script_choices = discover_model_scripts()
    parser = argparse.ArgumentParser(description='Draw Grad-CAM for a Knee OA checkpoint.')
    parser.add_argument('--model', default='ResNet_layer3+MECS+Loss4.py', choices=script_choices)
    parser.add_argument('--checkpoint', default='', help='Optional explicit checkpoint path. Defaults to the known checkpoint for the selected script.')
    parser.add_argument('--target-layer', default='layer4', help='Layer/module path to visualize, e.g. layer4, mecs, gcsa, or mdfa.')
    parser.add_argument('--target-class', type=int, default=None, help='Optional target class index. Defaults to predicted class.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--image-size', type=int, default=int(os.getenv('KNEE_IMAGE_SIZE', '224')))
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    parser.add_argument('--data-root', default='', help='Optional Knee data root override. Defaults to KNEE_DATA_ROOT or PROJECT_ROOT/Knee_Osteoarthritis.')
    parser.add_argument('--test-dir', default='', help='Optional explicit test directory override.')
    parser.add_argument('--class-name', default='', help='Optional true-label class filter. Supports both folder names and display names.')
    parser.add_argument('--sample-index', type=int, default=0, help='Sample index inside the filtered test set.')
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--list-classes', action='store_true')
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument('--image', default='', help='Optional explicit image path.')
    image_group.add_argument('--relative-path', default='', help='Optional relative path under test dir, e.g. 2/xxx.png')
    return parser.parse_args()


def build_eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def resolve_test_dir(args: argparse.Namespace) -> Path:
    if args.test_dir:
        return Path(args.test_dir).expanduser().resolve()
    data_root = Path(args.data_root or os.getenv('KNEE_DATA_ROOT', str(PROJECT_ROOT / 'Knee_Osteoarthritis')))
    return (data_root / 'test').resolve()


def load_knee_module(script_path: Path):
    return load_script_module(script_path, prefix='knee_gradcam')


def build_model_from_module(module, num_classes: int):
    if hasattr(module, 'build_model'):
        return module.build_model(num_classes)
    for class_name in ('CustomResNet50MECS', 'CustomResNet50GCSA', 'CustomResNet50MDFA'):
        cls = getattr(module, class_name, None)
        if cls is not None:
            return cls(num_classes=num_classes)

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


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    default_name = DEFAULT_CHECKPOINTS.get(args.model)
    if not default_name:
        raise ValueError(f'No default checkpoint mapping for {args.model}. Please pass --checkpoint.')
    return (THIS_DIR / default_name).resolve()


def build_test_records(test_dir: Path, class_names: List[str]) -> Dict[str, object]:
    dataset = datasets.ImageFolder(str(test_dir))
    rows = []
    for path, label in dataset.samples:
        display_name = class_names[label] if label < len(class_names) else dataset.classes[label]
        rows.append({
            'image_path': path,
            'label_index': int(label),
            'folder_name': dataset.classes[label],
            'label_name': display_name,
            'relative_path': str(Path(path).relative_to(test_dir)),
        })
    return {
        'dataset': dataset,
        'class_names': class_names,
        'rows': rows,
    }


def resolve_image_source(args: argparse.Namespace, test_dir: Path, records: Dict[str, object]) -> Dict[str, object]:
    rows = list(records['rows'])

    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f'Image path not found: {image_path}')
        return {'image_path': str(image_path), 'display_name': image_path.stem, 'true_label_name': None, 'true_label_index': None}

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
        return {'image_path': str(image_path), 'display_name': image_path.stem, 'true_label_name': None, 'true_label_index': None}

    if args.class_name:
        target_class = args.class_name.strip().lower()
        rows = [
            row for row in rows
            if str(row['label_name']).lower() == target_class or str(row['folder_name']).lower() == target_class
        ]

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


def main() -> None:
    args = parse_args()
    if args.list_models:
        print('Available Knee model scripts:')
        for name in discover_model_scripts():
            print(f'  {name}')
        return

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = (THIS_DIR / args.model).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f'Model script not found: {script_path}')
    module = load_knee_module(script_path)
    set_seed = getattr(module, 'set_seed', None)
    if callable(set_seed):
        set_seed(getattr(module, 'SEED', 1234))

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
    checkpoint_path = resolve_checkpoint_path(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    model = build_model_from_module(module, num_classes=len(class_names)).to(device)
    load_knee_checkpoint_states(checkpoint_path, model, device)
    model.eval()

    target_root = resolve_target_root(model, args.target_layer)
    target_module = find_last_conv(target_root)
    if target_module is None:
        raise RuntimeError('No Conv2d layer found under target-layer.')

    transform = build_eval_transform(args.image_size)
    pil_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = extract_logits(model(input_tensor))
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        pred_conf = float(probs[0, pred_idx].item())

    target_class = pred_idx if args.target_class is None else int(args.target_class)
    if target_class < 0 or target_class >= len(class_names):
        raise ValueError(f'target-class out of range: {target_class}')

    gradcam = GradCAM(model, target_module)
    try:
        with torch.enable_grad():
            cam, logits = gradcam.generate(input_tensor, class_idx=target_class)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(probs.argmax(dim=1).item())
            pred_conf = float(probs[0, pred_idx].item())
    finally:
        gradcam.remove()

    original = tensor_to_pil(input_tensor.squeeze(0), mean=MEAN, std=STD).resize((args.image_size, args.image_size), resample=BILINEAR)
    heatmap, overlay = build_heatmap_images(cam, original, alpha=args.alpha)

    pred_label_name = class_names[pred_idx]
    target_label_name = class_names[target_class]
    info_lines = [
        f'model={script_path.stem} | checkpoint={checkpoint_path.name}',
        f'image={image_path.name} | target_layer={args.target_layer}',
        f'pred={pred_label_name} ({pred_idx}) | confidence={pred_conf:.4f} | target_class={target_label_name} ({target_class})',
        f"true={image_source['true_label_name']} ({image_source['true_label_index']})" if image_source['true_label_name'] is not None else 'true=unknown (custom image)',
    ]
    panel = compose_panel(original, heatmap, overlay, info_lines)

    base_name = sanitize_filename('_'.join([script_path.stem, image_source['display_name'], args.target_layer.replace('.', '-')] ))
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
    print(f'Prediction: {pred_label_name} ({pred_idx}) | confidence={pred_conf:.4f}')
    if image_source['true_label_name'] is not None:
        print(f"True label: {image_source['true_label_name']} ({image_source['true_label_index']})")
    print(f'Target class for CAM: {target_label_name} ({target_class})')
    print(f'Saved overlay: {overlay_path}')
    print(f'Saved heatmap: {heatmap_path}')
    print(f'Saved panel: {panel_path}')


if __name__ == '__main__':
    main()