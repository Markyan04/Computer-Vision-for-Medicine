#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for Knee OA Grad-CAM visualizations powered by pytorch-grad-cam."""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from gradcam_shared import (
    BILINEAR,
    extract_logits,
    find_last_conv,
    resolve_target_root,
    tensor_to_pil,
)

try:
    from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, HiResCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _PYTORCH_GRAD_CAM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local env
    EigenCAM = None
    GradCAM = None
    GradCAMPlusPlus = None
    HiResCAM = None
    show_cam_on_image = None
    ClassifierOutputTarget = None
    _PYTORCH_GRAD_CAM_IMPORT_ERROR = exc


CAM_METHOD_CHOICES = ('gradcam', 'gradcam++', 'hirescam', 'eigencam')
_CAM_METHODS = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'hirescam': HiResCAM,
    'eigencam': EigenCAM,
}


def ensure_pytorch_grad_cam() -> None:
    if _PYTORCH_GRAD_CAM_IMPORT_ERROR is None:
        return
    raise ImportError(
        'pytorch-grad-cam is not installed in the current environment. '
        'Install it with: pip install grad-cam opencv-python'
    ) from _PYTORCH_GRAD_CAM_IMPORT_ERROR


def resolve_checkpoint_path(
    explicit_path: str,
    script_name: str,
    default_checkpoints: Dict[str, str],
    base_dir: Path,
) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    default_name = default_checkpoints.get(script_name)
    if not default_name:
        raise ValueError(
            f'No default checkpoint mapping for {script_name}. Please pass an explicit checkpoint path.'
        )

    candidates = [
        (base_dir / 'checkpoints' / default_name).resolve(),
        (base_dir / default_name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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


def _instantiate_cam_engine(
    method_name: str,
    model: torch.nn.Module,
    target_module: torch.nn.Module,
):
    ensure_pytorch_grad_cam()
    cam_cls = _CAM_METHODS.get(method_name)
    if cam_cls is None:
        raise ValueError(f'Unsupported CAM method: {method_name}')

    kwargs = {
        'model': model,
        'target_layers': [target_module],
    }
    try:
        return cam_cls(**kwargs)
    except TypeError:
        kwargs['use_cuda'] = next(model.parameters()).is_cuda
        return cam_cls(**kwargs)


def _run_cam_engine(
    cam_engine,
    input_tensor: torch.Tensor,
    class_idx: int,
    aug_smooth: bool,
    eigen_smooth: bool,
):
    ensure_pytorch_grad_cam()
    targets = [ClassifierOutputTarget(int(class_idx))]
    kwargs = {
        'input_tensor': input_tensor,
        'targets': targets,
    }
    try:
        return cam_engine(
            **kwargs,
            aug_smooth=bool(aug_smooth),
            eigen_smooth=bool(eigen_smooth),
        )
    except TypeError:
        return cam_engine(**kwargs)


def build_cam_images(
    method_name: str,
    model: torch.nn.Module,
    target_module: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    image_size: int,
    alpha: float,
    mean: Sequence[float],
    std: Sequence[float],
    aug_smooth: bool = False,
    eigen_smooth: bool = False,
):
    ensure_pytorch_grad_cam()
    cam_engine = _instantiate_cam_engine(method_name, model, target_module)
    try:
        grayscale_cam = _run_cam_engine(
            cam_engine,
            input_tensor=input_tensor,
            class_idx=class_idx,
            aug_smooth=aug_smooth,
            eigen_smooth=eigen_smooth,
        )
    finally:
        release = getattr(getattr(cam_engine, 'activations_and_grads', None), 'release', None)
        if callable(release):
            release()

    mask = np.asarray(grayscale_cam[0], dtype=np.float32)
    mask = np.clip(mask, 0.0, 1.0)

    original = tensor_to_pil(input_tensor.squeeze(0), mean=mean, std=std).resize(
        (image_size, image_size),
        resample=BILINEAR,
    )
    original_np = np.asarray(original).astype(np.float32) / 255.0

    heatmap_np = show_cam_on_image(original_np, mask, use_rgb=True, image_weight=0.0)
    overlay_weight = max(0.0, min(1.0, 1.0 - float(alpha)))
    overlay_np = show_cam_on_image(
        original_np,
        mask,
        use_rgb=True,
        image_weight=overlay_weight,
    )
    heatmap = Image.fromarray(heatmap_np)
    overlay = Image.fromarray(overlay_np)
    return original, heatmap, overlay, mask
