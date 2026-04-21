#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export Grad-CAM visualizations for every Alzheimer MRI image of a chosen class."""

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

from alzheimer_mri_loss_experiment_common import load_checkpoint_states, set_seed, SEED  # noqa: E402
from draw_gradcam_pytorch_grad_cam import (  # noqa: E402
    CHECKPOINT_DIR,
    MEAN,
    STD,
    build_eval_transform,
    build_split_records,
    discover_model_scripts,
    resolve_data_root,
)
from gradcam_shared import compose_panel, extract_logits, load_script_module, sanitize_filename, tensor_to_pil  # noqa: E402
from pytorch_grad_cam_shared import (  # noqa: E402
    CAM_METHOD_CHOICES,
    build_cam_images,
    ensure_pytorch_grad_cam,
    predict,
    resolve_target_module,
)
from gradcam_shared import resolve_device  # noqa: E402
from alzheimer_mri_loss_experiment_common import sanitize_run_tag  # noqa: E402


OUTPUT_ROOT = THIS_DIR / "gradcam_class_exports_pytorch_grad_cam"


def parse_args() -> argparse.Namespace:
    script_choices = discover_model_scripts()
    parser = argparse.ArgumentParser(
        description="Export Grad-CAM visualizations for all Alzheimer MRI images of a chosen class."
    )
    parser.add_argument("--class-name", required=True, help="True-label class name to filter.")
    parser.add_argument("--model", default="ResNet_layer3+MECS.py", choices=script_choices)
    parser.add_argument("--loss", default="dast", help="Loss suffix used in checkpoint naming, e.g. 'ce' or 'dast'.")
    parser.add_argument("--run-tag", default="", help="Optional run tag used in checkpoint naming.")
    parser.add_argument("--checkpoint", default="", help="Optional explicit checkpoint path.")
    parser.add_argument(
        "--target-layer",
        default="inserted_module",
        help="Layer/module path to visualize, e.g. inserted_module, inserted_module.post_conv, layer4.",
    )
    parser.add_argument("--target-class", type=int, default=None, help="Optional target class index. Overrides --cam-on.")
    parser.add_argument("--cam-on", default="true", choices=["pred", "true"], help="Which class to explain when --target-class is not provided. Default: true.")
    parser.add_argument("--cam-method", default="gradcam++", choices=CAM_METHOD_CHOICES, help="pytorch-grad-cam method. Default: gradcam++.")
    parser.add_argument("--aug-smooth", action="store_true", help="Enable test-time augmentation smoothing if supported.")
    parser.add_argument("--eigen-smooth", action="store_true", help="Enable eigen smoothing if supported.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--image-size", type=int, default=int(os.getenv("ALZHEIMER_IMAGE_SIZE", "224")))
    parser.add_argument("--alpha", type=float, default=0.35, help="Heatmap opacity. Default: 0.35.")
    parser.add_argument("--data-root", default="", help="Optional ALZHEIMER data root override.")
    parser.add_argument("--test-ratio", type=float, default=float(os.getenv("ALZHEIMER_TEST_RATIO", "0.2")))
    parser.add_argument("--val-ratio", type=float, default=float(os.getenv("ALZHEIMER_VAL_RATIO", "0.1")))
    parser.add_argument("--split", default="test", choices=["train", "valid", "test", "all"], help="Dataset split to export from. Default: test.")
    parser.add_argument("--output-dir", default="", help="Optional output directory. Defaults to a timestamped folder under gradcam_class_exports_pytorch_grad_cam/.")
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum number of images to export. Use <=0 for all. Default: 0.")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-classes", action="store_true")
    return parser.parse_args()


def resolve_checkpoint_path(args: argparse.Namespace, script_stem: str) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    run_tag = sanitize_run_tag(args.run_tag)
    suffix = f"_{run_tag}" if run_tag else ""
    return (CHECKPOINT_DIR / f"best_{script_stem}_{args.loss}{suffix}.pt").resolve()


def resolve_output_dir(raw: str, class_name: str, model_stem: str) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_class = sanitize_filename(class_name)
    safe_model = sanitize_filename(model_stem)
    return (OUTPUT_ROOT / f"{safe_model}_{safe_class}_{timestamp}").resolve()


def collect_class_records(records: Dict[str, object], split_name: str, class_name: str) -> List[Dict[str, object]]:
    target = class_name.strip().lower()
    return [
        row for row in records["rows"][split_name]
        if str(row["label_name"]).strip().lower() == target
    ]


def write_summary(summary_rows: Sequence[Dict[str, object]], output_dir: Path) -> Optional[Path]:
    if not summary_rows:
        return None

    preferred = [
        "image_id", "true_label", "image_path", "split", "cam_method", "cam_on",
        "prediction", "prediction_confidence", "cam_target",
        "original_path", "heatmap_path", "overlay_path", "panel_path",
    ]
    seen: List[str] = []
    for row in summary_rows:
        for key in row.keys():
            if key not in seen:
                seen.append(key)
    fieldnames = [field for field in preferred if field in seen]
    fieldnames.extend(field for field in seen if field not in fieldnames)

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return summary_path


def main() -> None:
    args = parse_args()
    if args.list_models:
        print("Available Alzheimer MRI model scripts:")
        for name in discover_model_scripts():
            print(f"  {name}")
        return

    ensure_pytorch_grad_cam()
    set_seed(SEED)
    data_root = resolve_data_root(args)
    records = build_split_records(data_root, test_ratio=args.test_ratio, val_ratio=args.val_ratio)
    class_names = list(records["class_names"])

    if args.list_classes:
        print("Available Alzheimer MRI classes:")
        for name in class_names:
            print(f"  {name}")
        return

    class_lookup = {name.lower(): idx for idx, name in enumerate(class_names)}
    target_class_name = args.class_name.strip().lower()
    if target_class_name not in class_lookup:
        print(f"Data root: {data_root}")
        print(f"Unknown class-name: {args.class_name}")
        print("Available classes: " + ", ".join(class_names))
        return

    script_path = (THIS_DIR / args.model).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Model script not found: {script_path}")
    module = load_script_module(script_path, prefix="alz_mri_gradcam_class_pgc")
    build_model = getattr(module, "build_model", None)
    if build_model is None:
        raise AttributeError(f"build_model() not found in {script_path.name}")

    checkpoint_path = resolve_checkpoint_path(args, script_path.stem)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    output_dir = resolve_output_dir(args.output_dir, target_class_name, script_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(len(class_names)).to(device)
    load_checkpoint_states(checkpoint_path, model, device, criterion=None)
    model.eval()

    target_module = resolve_target_module(
        model,
        args.target_layer,
        hint="Try inserted_module, inserted_module.post_conv, layer4, layer3, or layer2.",
    )

    transform = build_eval_transform(args.image_size)
    candidate_rows = collect_class_records(records, args.split, target_class_name)

    print(f"Device: {device}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Data root: {data_root}")
    print(f"Model script: {script_path.name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split} | split size: {len(records['rows'][args.split])}")
    print(f"Candidate class: {target_class_name} | candidates in split: {len(candidate_rows)}")
    print(f"CAM method: {args.cam_method} | cam_on={args.cam_on} | alpha={args.alpha:.2f}")
    if args.aug_smooth or args.eigen_smooth:
        print(f"CAM smoothing: aug_smooth={args.aug_smooth}, eigen_smooth={args.eigen_smooth}")
    print(f"Target layer: {args.target_layer} -> {target_module.__class__.__name__}")
    print(f"Output dir: {output_dir}")

    if not candidate_rows:
        print("No matching samples found. Nothing was exported.")
        return

    summary_rows: List[Dict[str, object]] = []
    exported = 0

    for row in candidate_rows:
        image_path = Path(str(row["image_path"]))
        pil_image = Image.open(image_path).convert("RGB")
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        pred_idx, pred_conf = predict(model, input_tensor, logits_extractor=extract_logits)
        true_idx = int(row["label_index"])

        if args.target_class is not None:
            cam_target_idx = int(args.target_class)
            cam_target_source = "explicit"
        elif args.cam_on == "true":
            cam_target_idx = true_idx
            cam_target_source = "true"
        else:
            cam_target_idx = pred_idx
            cam_target_source = "pred"

        if cam_target_idx < 0 or cam_target_idx >= len(class_names):
            raise ValueError(f"target-class out of range: {cam_target_idx}")

        original, heatmap, overlay, _ = build_cam_images(
            method_name=args.cam_method,
            model=model,
            target_module=target_module,
            input_tensor=input_tensor,
            class_idx=cam_target_idx,
            image_size=args.image_size,
            alpha=args.alpha,
            original_from_tensor=lambda t: tensor_to_pil(t, mean=MEAN, std=STD),
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )

        pred_label_name = class_names[pred_idx]
        true_label_name = class_names[true_idx]
        cam_target_name = class_names[cam_target_idx]
        image_id = str(row["display_name"])

        info_lines = [
            f"image_id={image_id} | split={row['split']} | true={true_label_name}",
            f"pred={pred_label_name} ({pred_conf:.4f}) | cam_target={cam_target_name}",
            f"cam_method={args.cam_method} | cam_on={cam_target_source} | layer={args.target_layer}",
        ]
        if args.aug_smooth or args.eigen_smooth:
            info_lines.append(f"aug_smooth={args.aug_smooth} | eigen_smooth={args.eigen_smooth}")
        panel = compose_panel(original, heatmap, overlay, info_lines)

        stem = sanitize_filename(
            f"{image_id}_true-{true_label_name}_pred-{pred_label_name}_{args.target_layer.replace('.', '-')}_{args.cam_method.replace('+', 'plus')}_camon-{cam_target_source}"
        )
        original_path = output_dir / f"{stem}_original.png"
        heatmap_path = output_dir / f"{stem}_heatmap.png"
        overlay_path = output_dir / f"{stem}_overlay.png"
        panel_path = output_dir / f"{stem}_panel.png"

        original.save(original_path)
        heatmap.save(heatmap_path)
        overlay.save(overlay_path)
        panel.save(panel_path)

        summary_rows.append({
            "image_id": image_id,
            "true_label": true_label_name,
            "image_path": str(image_path),
            "split": row["split"],
            "cam_method": args.cam_method,
            "cam_on": cam_target_source,
            "prediction": pred_label_name,
            "prediction_confidence": f"{pred_conf:.6f}",
            "cam_target": cam_target_name,
            "original_path": str(original_path),
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
            "panel_path": str(panel_path),
        })
        exported += 1
        print(f"[{exported}] exported image_id={image_id} | true={true_label_name} | pred={pred_label_name}")

        if args.max_samples > 0 and exported >= args.max_samples:
            break

    summary_path = write_summary(summary_rows, output_dir)
    print(f"Exported {len(summary_rows)} sample(s) to: {output_dir}")
    if summary_path is not None:
        print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
