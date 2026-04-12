#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Draw Grad-CAM visualizations for HAM10000 checkpoints."""

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from ham10000_loss_experiment_common import (  # noqa: E402
    SEED,
    ISICDataset,
    _build_valid_dataframe,
    load_checkpoint_states,
    sanitize_run_tag,
    set_seed,
)
from sklearn.model_selection import train_test_split  # noqa: E402


CHECKPOINT_DIR = THIS_DIR / 'checkpoints'
OUTPUT_DIR = THIS_DIR / 'gradcam_outputs'
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR


def discover_model_scripts() -> List[str]:
    scripts = []
    for path in sorted(THIS_DIR.glob('ResNet*.py')):
        if path.name.endswith('_warmstart.py'):
            continue
        scripts.append(path.name)
    return scripts


def parse_args() -> argparse.Namespace:
    script_choices = discover_model_scripts()
    parser = argparse.ArgumentParser(description='Draw Grad-CAM for a HAM10000 model checkpoint.')
    parser.add_argument(
        '--model',
        default='ResNet_layer3+MECS.py',
        choices=script_choices,
        help='Model script to load. Default: ResNet_layer3+MECS.py',
    )
    parser.add_argument(
        '--loss',
        default='ce',
        help="Loss suffix used in the default checkpoint name, e.g. 'ce' or 'dast'.",
    )
    parser.add_argument(
        '--run-tag',
        default='',
        help='Optional run tag used in checkpoint naming, e.g. dast_tune_tau1_gamma1p5.',
    )
    parser.add_argument(
        '--checkpoint',
        default='',
        help='Optional explicit checkpoint path. Overrides --loss/--run-tag resolution.',
    )
    parser.add_argument(
        '--target-layer',
        default='layer4',
        help='Layer root or module path to visualize. Examples: layer4, layer3, inserted_module, inserted_module.post_conv.',
    )
    parser.add_argument(
        '--target-class',
        type=int,
        default=None,
        help='Optional target class index for Grad-CAM. Defaults to the predicted class.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help="Inference device. Default: auto.",
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=int(os.getenv('HAM10000_IMAGE_SIZE', '224')),
        help='Resize / center-crop size. Default reads HAM10000_IMAGE_SIZE or falls back to 224.',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.45,
        help='Overlay alpha for the heatmap. Default: 0.45.',
    )
    parser.add_argument(
        '--output-dir',
        default=str(OUTPUT_DIR),
        help='Directory for saved visualizations.',
    )
    parser.add_argument(
        '--data-dir',
        default='',
        help='Optional HAM10000 data directory override. Defaults to HAM10000_DATA_DIR or PROJECT_ROOT/ISIC.',
    )
    parser.add_argument(
        '--split',
        default='test',
        choices=['train', 'valid', 'test', 'all'],
        help='Dataset split used when selecting by sample index. Default: test.',
    )
    parser.add_argument(
        '--sample-index',
        type=int,
        default=0,
        help='Sample index inside the chosen split when no image path / image_id is provided. Default: 0.',
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available model scripts and exit.',
    )

    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        '--image',
        default='',
        help='Optional explicit image path.',
    )
    image_group.add_argument(
        '--image-id',
        default='',
        help='Optional HAM10000 image_id from metadata.',
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == 'cpu':
        return torch.device('cpu')
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_model_builder(script_path: Path):
    module_name = 'ham_gradcam_' + re.sub(r'[^0-9A-Za-z_]+', '_', script_path.stem)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load model script: {script_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_model = getattr(module, 'build_model', None)
    if build_model is None:
        raise AttributeError(f'build_model() not found in: {script_path}')
    return build_model


def resolve_checkpoint_path(args: argparse.Namespace, script_stem: str) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()

    run_tag = sanitize_run_tag(args.run_tag)
    suffix = f'_{run_tag}' if run_tag else ''
    return CHECKPOINT_DIR / f'best_{script_stem}_{args.loss}{suffix}.pt'


def build_split_records(data_dir: Path) -> Tuple[List[str], Dict[str, List[Dict[str, object]]]]:
    valid_df = _build_valid_dataframe(data_dir)
    base_dataset = ISICDataset(valid_df, transform=None)
    class_names = list(base_dataset.labels)

    targets = np.array(base_dataset.targets)
    indices = np.arange(len(base_dataset))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets,
        random_state=SEED,
    )
    train_targets = targets[train_idx]
    train_idx, valid_idx = train_test_split(
        train_idx,
        test_size=0.1,
        stratify=train_targets,
        random_state=SEED,
    )

    split_map = {
        'train': train_idx.tolist(),
        'valid': valid_idx.tolist(),
        'test': test_idx.tolist(),
        'all': indices.tolist(),
    }

    records: Dict[str, List[Dict[str, object]]] = {}
    for split_name, split_indices in split_map.items():
        rows: List[Dict[str, object]] = []
        for idx in split_indices:
            rows.append({
                'base_index': int(idx),
                'image_id': str(base_dataset.df.iloc[idx]['image_id']),
                'dx': str(base_dataset.df.iloc[idx]['dx']),
                'label_index': int(base_dataset.targets[idx]),
                'image_path': str(base_dataset.img_paths[idx]),
            })
        records[split_name] = rows

    return class_names, records


def resolve_image_source(
    args: argparse.Namespace,
    class_names: Sequence[str],
    split_records: Dict[str, List[Dict[str, object]]],
) -> Dict[str, object]:
    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f'Image path not found: {image_path}')
        return {
            'image_path': str(image_path),
            'image_id': image_path.stem,
            'true_label_name': None,
            'true_label_index': None,
            'source_split': 'custom',
        }

    if args.image_id:
        for split_name, rows in split_records.items():
            for row in rows:
                if row['image_id'] == args.image_id:
                    return {
                        'image_path': row['image_path'],
                        'image_id': row['image_id'],
                        'true_label_name': row['dx'],
                        'true_label_index': row['label_index'],
                        'source_split': split_name,
                    }
        raise ValueError(f'image_id not found in HAM10000 metadata: {args.image_id}')

    rows = split_records[args.split]
    if not rows:
        raise RuntimeError(f'No samples available in split: {args.split}')
    if args.sample_index < 0 or args.sample_index >= len(rows):
        raise IndexError(
            f'sample-index out of range for split {args.split}: {args.sample_index} (size={len(rows)})'
        )

    row = rows[args.sample_index]
    return {
        'image_path': row['image_path'],
        'image_id': row['image_id'],
        'true_label_name': row['dx'],
        'true_label_index': row['label_index'],
        'source_split': args.split,
    }


def resolve_target_root(model: nn.Module, target_layer: str) -> nn.Module:
    module: nn.Module = model
    for token in target_layer.split('.'):
        if not token:
            continue
        if token.isdigit():
            module = module[int(token)]
        else:
            module = getattr(module, token)
    return module


def find_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    if isinstance(module, nn.Conv2d):
        return module
    children = list(module.children())
    for child in reversed(children):
        found = find_last_conv(child)
        if found is not None:
            return found
    return None


class GradCAM:
    def __init__(self, model: nn.Module, target_module: nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = [
            target_module.register_forward_hook(self._forward_hook),
            target_module.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> Tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits, _ = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError('Grad-CAM hooks did not capture activations/gradients.')

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)
        return cam.astype(np.float32), logits.detach()


def tensor_to_pil(input_tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor(MEAN, dtype=input_tensor.dtype).view(3, 1, 1)
    std = torch.tensor(STD, dtype=input_tensor.dtype).view(3, 1, 1)
    image = input_tensor.detach().cpu() * std + mean
    image = image.clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def build_heatmap_images(cam: np.ndarray, base_image: Image.Image, alpha: float) -> Tuple[Image.Image, Image.Image]:
    cam_uint8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    gray = Image.fromarray(cam_uint8, mode='L').resize(base_image.size, resample=BILINEAR)
    heatmap = ImageOps.colorize(gray, black='#00007F', mid='#FFFF00', white='#FF0000').convert('RGB')
    overlay = Image.blend(base_image.convert('RGB'), heatmap, alpha=max(0.0, min(alpha, 1.0)))
    return heatmap, overlay


def compose_panel(
    original: Image.Image,
    heatmap: Image.Image,
    overlay: Image.Image,
    info_lines: Sequence[str],
) -> Image.Image:
    margin = 20
    title_gap = 26
    header_height = 110
    panel_width = original.width * 3 + margin * 4
    panel_height = header_height + original.height + title_gap + margin

    canvas = Image.new('RGB', (panel_width, panel_height), color='white')
    draw = ImageDraw.Draw(canvas)

    y = 12
    for line in info_lines:
        draw.text((margin, y), line, fill='black')
        y += 18

    positions = [margin, margin * 2 + original.width, margin * 3 + original.width * 2]
    titles = ['Original', 'Grad-CAM Heatmap', 'Overlay']
    images = [original.convert('RGB'), heatmap, overlay]

    for x, title, image in zip(positions, titles, images):
        draw.text((x, header_height - 24), title, fill='black')
        canvas.paste(image, (x, header_height))

    return canvas


def sanitize_filename(text: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', text.strip())
    return cleaned.strip('._') or 'sample'


def main() -> None:
    args = parse_args()
    if args.list_models:
        print('Available HAM10000 model scripts:')
        for name in discover_model_scripts():
            print(f'  {name}')
        return

    set_seed(SEED)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir or os.getenv('HAM10000_DATA_DIR', str(PROJECT_ROOT / 'ISIC'))).resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names, split_records = build_split_records(data_dir)
    image_source = resolve_image_source(args, class_names=class_names, split_records=split_records)
    image_path = Path(str(image_source['image_path']))

    script_path = (THIS_DIR / args.model).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f'Model script not found: {script_path}')
    script_stem = script_path.stem
    checkpoint_path = resolve_checkpoint_path(args, script_stem)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    build_model = load_model_builder(script_path)
    model = build_model(len(class_names)).to(device)
    load_checkpoint_states(checkpoint_path, model, device, criterion=None)
    model.eval()

    target_root = resolve_target_root(model, args.target_layer)
    target_module = find_last_conv(target_root)
    if target_module is None:
        raise RuntimeError(
            f'No Conv2d layer found under target-layer={args.target_layer}. '
            'Try layer4, layer3, layer2, inserted_module, or inserted_module.post_conv.'
        )

    transform = build_eval_transform(args.image_size)
    pil_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    gradcam = GradCAM(model, target_module)
    try:
        with torch.enable_grad():
            with torch.no_grad():
                logits, _ = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(probs.argmax(dim=1).item())
            pred_conf = float(probs[0, pred_idx].item())

            target_class = pred_idx if args.target_class is None else int(args.target_class)
            if target_class < 0 or target_class >= len(class_names):
                raise ValueError(f'target-class out of range: {target_class}')

            cam, logits = gradcam.generate(input_tensor, class_idx=target_class)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(probs.argmax(dim=1).item())
            pred_conf = float(probs[0, pred_idx].item())
    finally:
        gradcam.remove()

    original = tensor_to_pil(input_tensor.squeeze(0)).resize((args.image_size, args.image_size), resample=BILINEAR)
    heatmap, overlay = build_heatmap_images(cam, original, alpha=args.alpha)

    true_label_name = image_source['true_label_name']
    true_label_index = image_source['true_label_index']
    pred_label_name = class_names[pred_idx]
    target_label_name = class_names[target_class]

    info_lines = [
        f'model={script_stem} | checkpoint={checkpoint_path.name}',
        f'image_id={image_source["image_id"]} | split={image_source["source_split"]} | target_layer={args.target_layer}',
        f'pred={pred_label_name} ({pred_idx}) | confidence={pred_conf:.4f} | target_class={target_label_name} ({target_class})',
        f'true={true_label_name} ({true_label_index})' if true_label_name is not None else 'true=unknown (custom image)',
    ]
    panel = compose_panel(original, heatmap, overlay, info_lines)

    run_tag = sanitize_run_tag(args.run_tag)
    base_name_parts = [script_stem, args.loss]
    if run_tag:
        base_name_parts.append(run_tag)
    base_name_parts.append(str(image_source['image_id']))
    base_name_parts.append(args.target_layer.replace('.', '-'))
    base_name = sanitize_filename('_'.join(base_name_parts))

    overlay_path = output_dir / f'{base_name}_overlay.png'
    heatmap_path = output_dir / f'{base_name}_heatmap.png'
    panel_path = output_dir / f'{base_name}_panel.png'

    overlay.save(overlay_path)
    heatmap.save(heatmap_path)
    panel.save(panel_path)

    print(f'Device: {device}')
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f'CUDA: {torch.cuda.get_device_name(0)}')
    print(f'Model script: {script_path.name}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Image: {image_path}')
    print(f'Target layer: {args.target_layer} -> {target_module.__class__.__name__}')
    print(f'Prediction: {pred_label_name} ({pred_idx}) | confidence={pred_conf:.4f}')
    if true_label_name is not None:
        print(f'True label: {true_label_name} ({true_label_index})')
    print(f'Target class for CAM: {target_label_name} ({target_class})')
    print(f'Saved overlay: {overlay_path}')
    print(f'Saved heatmap: {heatmap_path}')
    print(f'Saved panel: {panel_path}')


if __name__ == '__main__':
    main()