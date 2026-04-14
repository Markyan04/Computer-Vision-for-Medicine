#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared helpers for Grad-CAM scripts across datasets."""

import importlib.util
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F


BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR


def load_script_module(script_path: Path, prefix: str = 'gradcam_module'):
    module_name = prefix + '_' + re.sub(r'[^0-9A-Za-z_]+', '_', script_path.stem)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load script module: {script_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_logits(model_output):
    if isinstance(model_output, (tuple, list)):
        if not model_output:
            raise RuntimeError('Model output tuple/list is empty.')
        return model_output[0]
    return model_output


def resolve_device(device_name: str):
    if device_name == 'cpu':
        return torch.device('cpu')
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        logits = extract_logits(self.model(input_tensor))
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


def tensor_to_pil(input_tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> Image.Image:
    mean_tensor = torch.tensor(mean, dtype=input_tensor.dtype).view(len(mean), 1, 1)
    std_tensor = torch.tensor(std, dtype=input_tensor.dtype).view(len(std), 1, 1)
    image = input_tensor.detach().cpu() * std_tensor + mean_tensor
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
    header_height = 110
    panel_width = original.width * 3 + margin * 4
    panel_height = header_height + original.height + margin

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