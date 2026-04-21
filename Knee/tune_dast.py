#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Grid search DAST tau/gamma for Knee baseline from strict to loose."""

import argparse
import csv
import math
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


THIS_DIR = Path(__file__).resolve().parent
LOG_DIR = THIS_DIR / "logs"
BATCH_LOG_DIR = THIS_DIR / "batch_logs"
DEFAULT_SCRIPT_ITEMS: Tuple[Tuple[str, str], ...] = (
    ("ResNet_baseline+Loss4.py", "baseline"),
)
SCRIPT_TOKEN_MAP = {
    "baseline": ("ResNet_baseline+Loss4.py", "baseline"),
    "baseline+dast": ("ResNet_baseline+Loss4.py", "baseline"),
    "baseline_dast": ("ResNet_baseline+Loss4.py", "baseline"),
    "resnet_baseline+loss4.py": ("ResNet_baseline+Loss4.py", "baseline"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune DAST tau/gamma for Knee baseline from strict to loose.",
    )
    parser.add_argument(
        "--taus",
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated tau values. Smaller tau is stricter. Default: '0.5,1.0,1.5,2.0'.",
    )
    parser.add_argument(
        "--gammas",
        default="2.0,1.5,1.0",
        help="Comma-separated gamma values. Higher gamma is stricter. Default: '2.0,1.5,1.0,0.0'.",
    )
    parser.add_argument(
        "--scripts",
        default="baseline",
        help="Comma-separated script tokens. Currently supports 'baseline' or 'ResNet_baseline+Loss4.py'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("KNEE_BATCH_SIZE", "32")),
        help="Training batch size. Default reads KNEE_BATCH_SIZE or falls back to 32.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for KNEE_EPOCHS.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for KNEE_NUM_WORKERS.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional override for KNEE_IMAGE_SIZE.",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=None,
        help="Optional override for KNEE_LR_BACKBONE.",
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=None,
        help="Optional override for KNEE_LR_HEAD.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Optional override for KNEE_WEIGHT_DECAY.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Optional override for KNEE_PATIENCE.",
    )
    parser.add_argument(
        "--early-delta",
        type=float,
        default=None,
        help="Optional override for KNEE_EARLY_DELTA.",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Optional override for KNEE_DATA_ROOT.",
    )
    parser.add_argument(
        "--tag-prefix",
        default="dast_tune",
        help="Prefix used in run tags and checkpoint names. Default: 'dast_tune'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned script/tau/gamma combinations without launching training.",
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("At least one numeric value is required.")
    return values


def float_slug(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def sanitize_run_tag(run_tag: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in run_tag.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-._")


def parse_scripts(raw: str) -> List[Tuple[str, str]]:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if not requested:
        raise ValueError("At least one script must be provided.")

    items: List[Tuple[str, str]] = []
    for token in requested:
        mapped = SCRIPT_TOKEN_MAP.get(token.lower())
        if mapped is None:
            raise ValueError(
                f"Unsupported script token: {token}. Supported values: baseline, baseline+dast, ResNet_baseline+Loss4.py"
            )
        items.append(mapped)
    return items


def build_run_tag(tag_prefix: str, script_alias: str, tau: float, gamma: float) -> str:
    return sanitize_run_tag(
        f"{script_alias}_{tag_prefix}_tau{float_slug(tau)}_gamma{float_slug(gamma)}"
    )


def build_env(args: argparse.Namespace, tau: float, gamma: float, run_tag: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["KNEE_BATCH_SIZE"] = str(args.batch_size)
    env["KNEE_DAST_TAU"] = str(tau)
    env["KNEE_DAST_GAMMA"] = str(gamma)
    env["KNEE_RUN_TAG"] = run_tag

    if args.epochs is not None:
        env["KNEE_EPOCHS"] = str(args.epochs)
    if args.num_workers is not None:
        env["KNEE_NUM_WORKERS"] = str(args.num_workers)
    if args.image_size is not None:
        env["KNEE_IMAGE_SIZE"] = str(args.image_size)
    if args.lr_backbone is not None:
        env["KNEE_LR_BACKBONE"] = str(args.lr_backbone)
    if args.lr_head is not None:
        env["KNEE_LR_HEAD"] = str(args.lr_head)
    if args.weight_decay is not None:
        env["KNEE_WEIGHT_DECAY"] = str(args.weight_decay)
    if args.patience is not None:
        env["KNEE_PATIENCE"] = str(args.patience)
    if args.early_delta is not None:
        env["KNEE_EARLY_DELTA"] = str(args.early_delta)
    if args.data_root:
        env["KNEE_DATA_ROOT"] = args.data_root

    return env


def safe_float(value: object, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def format_metric(value: object, digits: int = 4) -> str:
    parsed = safe_float(value)
    if math.isnan(parsed):
        return "n/a"
    return f"{parsed:.{digits}f}"


def load_summary_row(summary_path: Path) -> Optional[Dict[str, object]]:
    with open(summary_path, "r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    if not rows:
        return None
    return dict(rows[0])


def run_one_trial(
    script_name: str,
    script_alias: str,
    tau: float,
    gamma: float,
    env: Dict[str, str],
    run_tag: str,
    run_log_dir: Path,
) -> Dict[str, object]:
    run_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_log_dir / f"{run_tag}.log"

    header = (
        f"\n{'=' * 100}\n"
        f"RUN TAG: {run_tag}\n"
        f"SCRIPT: {script_name}\n"
        f"tau={tau}, gamma={gamma}\n"
        f"BATCH SIZE: {env.get('KNEE_BATCH_SIZE', '')}\n"
        f"START TIME: {datetime.now().isoformat(timespec='seconds')}\n"
        f"LOG FILE: {log_path}\n"
        f"{'=' * 100}\n"
    )
    print(header, end="")

    summary_path: Optional[Path] = None

    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write(header)
        fp.flush()

        proc = subprocess.Popen(
            [sys.executable, script_name],
            cwd=str(THIS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            fp.write(line)
            fp.flush()
            if line.startswith("Summary CSV saved: "):
                summary_path = Path(line.split(": ", 1)[1].strip())

        proc.wait()
        return_code = int(proc.returncode)

        footer = (
            f"\nEXIT CODE: {return_code}\n"
            f"END TIME: {datetime.now().isoformat(timespec='seconds')}\n"
            f"{'=' * 100}\n"
        )
        print(footer, end="")
        fp.write(footer)
        fp.flush()

    result: Dict[str, object] = {
        "script_name": script_name,
        "script_alias": script_alias,
        "run_tag": run_tag,
        "tau": tau,
        "gamma": gamma,
        "return_code": return_code,
        "batch_log_path": str(log_path),
        "summary_path": str(summary_path) if summary_path else "",
        "status": "failed",
    }

    if return_code != 0:
        result["error"] = f"training exited with code {return_code}"
        return result

    if summary_path is None or not summary_path.exists():
        result["error"] = "summary csv not found after training"
        return result

    row = load_summary_row(summary_path)
    if row is None:
        result["error"] = "no summary row found in summary csv"
        return result

    summary_status = str(row.get("status", "success"))
    row.update({key: value for key, value in result.items() if key != "status"})
    row["status"] = summary_status
    return row


def sort_success_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    success_rows = [row for row in rows if str(row.get("status", "")).lower() == "success"]
    return sorted(
        success_rows,
        key=lambda row: (
            str(row.get("script_alias", "")),
            -safe_float(row.get("best_valid_qwk"), default=float("-inf")),
            safe_float(row.get("best_valid_loss"), default=float("inf")),
            -safe_float(row.get("test_qwk"), default=float("-inf")),
            -safe_float(row.get("test_macro_f1"), default=float("-inf")),
        ),
    )


def print_ranked_results(rows: Sequence[Dict[str, object]]) -> None:
    ranked_rows = sort_success_rows(rows)
    if not ranked_rows:
        print("No successful trials to rank.")
        return

    print("\nTop successful trials by script (ranked by best_valid_qwk, best_valid_loss):")
    current_script = None
    for row in ranked_rows:
        script_alias = str(row.get("script_alias", ""))
        if script_alias != current_script:
            current_script = script_alias
            print(f"\n[{script_alias}]")
        print(
            f"tau={row.get('tau')} | gamma={row.get('gamma')} | "
            f"best_valid_qwk={format_metric(row.get('best_valid_qwk'))} | "
            f"best_valid_macro_f1={format_metric(row.get('best_valid_macro_f1'))} | "
            f"test_qwk={format_metric(row.get('test_qwk'))} | "
            f"test_macro_f1={format_metric(row.get('test_macro_f1'))} | "
            f"checkpoint={row.get('checkpoint_path', '')}"
        )


def write_results(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    preferred_fields = [
        "script_name",
        "script_alias",
        "run_tag",
        "tau",
        "gamma",
        "status",
        "return_code",
        "best_valid_qwk",
        "best_valid_macro_f1",
        "best_valid_loss",
        "test_qwk",
        "test_macro_f1",
        "test_acc",
        "test_mae",
        "trained_epochs",
        "best_epoch",
        "test_loss",
        "test_top1",
        "test_top2",
        "test_top3",
        "test_balanced_acc",
        "test_weighted_f1",
        "test_precision_macro",
        "test_recall_macro",
        "test_ovr_roc_auc_macro",
        "test_ovr_pr_auc_macro",
        "dast_tau",
        "dast_gamma",
        "checkpoint_path",
        "summary_path",
        "batch_log_path",
        "error",
    ]

    discovered_fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in discovered_fields:
                discovered_fields.append(key)

    fieldnames = [field for field in preferred_fields if field in discovered_fields]
    fieldnames.extend(field for field in discovered_fields if field not in fieldnames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    taus = parse_float_list(args.taus)
    gammas = parse_float_list(args.gammas)
    script_items = parse_scripts(args.scripts)

    if any(tau <= 0 for tau in taus):
        raise ValueError("All tau values must be > 0.")
    if any(gamma < 0 for gamma in gammas):
        raise ValueError("All gamma values must be >= 0.")

    combos = [
        (script_name, script_alias, tau, gamma)
        for script_name, script_alias in script_items
        for tau, gamma in product(taus, gammas)
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = BATCH_LOG_DIR / f"dast_tuning_{timestamp}"
    output_path = LOG_DIR / f"dast_tuning_{timestamp}.csv"

    print(f"Tuning scripts: {[item[0] for item in script_items]}")
    print(f"Trials: {len(combos)}")
    print(f"taus={taus}")
    print(f"gammas={gammas}")
    print(f"batch_size={args.batch_size}")
    if args.epochs is not None:
        print(f"epochs={args.epochs}")
    if args.lr_backbone is not None:
        print(f"lr_backbone={args.lr_backbone}")
    if args.lr_head is not None:
        print(f"lr_head={args.lr_head}")
    if args.data_root:
        print(f"data_root={args.data_root}")
    print(f"Batch logs dir: {run_log_dir}")
    print(f"Results CSV: {output_path}")

    if args.dry_run:
        print("\nPlanned trials:")
        for script_name, script_alias, tau, gamma in combos:
            run_tag = build_run_tag(args.tag_prefix, script_alias, tau, gamma)
            print(f"  script={script_name} | tau={tau} | gamma={gamma} | run_tag={run_tag}")
        return

    rows: List[Dict[str, object]] = []
    for index, (script_name, script_alias, tau, gamma) in enumerate(combos, start=1):
        run_tag = build_run_tag(args.tag_prefix, script_alias, tau, gamma)
        print(
            f"\n[{index}/{len(combos)}] Launching script={script_name} | "
            f"tau={tau}, gamma={gamma} | run_tag={run_tag}"
        )
        env = build_env(args, tau, gamma, run_tag)
        row = run_one_trial(
            script_name=script_name,
            script_alias=script_alias,
            tau=tau,
            gamma=gamma,
            env=env,
            run_tag=run_tag,
            run_log_dir=run_log_dir,
        )
        rows.append(row)

        status = str(row.get("status", "failed")).lower()
        if status == "success":
            print(
                "Trial result | "
                f"best_valid_qwk={format_metric(row.get('best_valid_qwk'))} | "
                f"test_qwk={format_metric(row.get('test_qwk'))} | "
                f"test_macro_f1={format_metric(row.get('test_macro_f1'))} | "
                f"checkpoint={row.get('checkpoint_path', '')}"
            )
        else:
            print(f"Trial failed | {row.get('error', 'unknown error')}")

        write_results(rows, output_path)

    print_ranked_results(rows)
    print(f"\nSaved tuning results to: {output_path}")

    success_count = len([row for row in rows if str(row.get("status", "")).lower() == "success"])
    print(f"Successful trials: {success_count}/{len(rows)}")


if __name__ == "__main__":
    main()
