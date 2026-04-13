"""
ERPDiff Ablation Runner — Dataset 1 & Dataset 2
================================================
运行所有消融实验（Exp 4–8）以及 Full Model baseline，覆盖 dataset1 和 dataset2。

实验清单（每个 dataset × 6 组 = 12 次完整 pretrain+finetune 流程）：
  full       : 完整 ERPDiff（所有组件启用）
  exp4_no_tb : w/o TDSA temporal bias  (--no-temporal-bias)
  exp5_no_ag : w/o GDCM alpha gate     (--no-alpha-gate)
  exp6_no_lc : w/o complementarity loss(--lambda-comp 0.0)
  exp7_no_bsd: w/o BSD, CE-only CLB    (--clb-bsd-beta 0.0)
  exp8_no_wu : w/o warmup unfreezing   (--warmup-epochs 0)

使用方式：
  python run_ablation_d1_d2.py
  python run_ablation_d1_d2.py --datasets dataset1
  python run_ablation_d1_d2.py --experiments full exp4_no_tb
  python run_ablation_d1_d2.py --epochs 300 --runs 3 --device cuda:0
  python run_ablation_d1_d2.py --resume        # 跳过已存在 finetune_report.json 的实验

注意：
  脚本会自动检测本目录中是否存在 erpdiff_train_modified.py；
  若不存在，则直接调用 erpdiff_train.py（需手动完成 --clb-bsd-beta 修改）。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# 实验矩阵定义
# ============================================================

# 每个实验配置：(name, description, extra_cli_args)
EXPERIMENT_CONFIGS: List[Tuple[str, str, List[str]]] = [
    (
        "full",
        "ERPDiff full model (all components enabled)",
        [],
    ),
    (
        "exp4_no_tb",
        "Ablation Exp4: w/o TDSA temporal bias",
        ["--no-temporal-bias"],
    ),
    (
        "exp5_no_ag",
        "Ablation Exp5: w/o GDCM alpha gate",
        ["--no-alpha-gate"],
    ),
    (
        "exp6_no_lc",
        "Ablation Exp6: w/o complementarity loss (lambda_comp=0)",
        ["--lambda-comp", "0.0"],
    ),
    (
        "exp7_no_bsd",
        "Ablation Exp7: w/o BSD self-distillation (CE-only CLB pretrain)",
        ["--clb-bsd-beta", "0.0"],
    ),
    (
        "exp8_no_wu",
        "Ablation Exp8: w/o warmup progressive unfreezing",
        ["--warmup-epochs", "0"],
    ),
    (
        "exp9_multi_sigma",
        "Ablation Exp9: multi-sigma temporal bias (per-head sigma)",
        ["--multi-sigma"],
    ),
]

EXPERIMENT_NAMES = [name for name, _, _ in EXPERIMENT_CONFIGS]
DATASET_CHOICES  = ["dataset1", "dataset2", "dataset3"]

# Output directory name for each experiment: ablation_<tag>_d<N>
# e.g.  full / dataset2  →  ablation_full_d2
#       exp5_no_ag / dataset1  →  ablation_alpha_gate_d1
EXP_DIR_TAG: Dict[str, str] = {
    "full":        "full",
    "exp4_no_tb":  "temporal_bias",
    "exp5_no_ag":  "alpha_gate",
    "exp6_no_lc":  "lambda_comp",
    "exp7_no_bsd": "clb_bsd",
    "exp8_no_wu":  "warmup",
    "exp9_multi_sigma": "multi_sigma",
}


def exp_out_name(exp_name: str, dataset: str) -> str:
    """Return the output directory name: ablation_<tag>_d<N>."""
    tag    = EXP_DIR_TAG.get(exp_name, exp_name)
    ds_num = dataset[-1]          # "dataset1" → "1"
    return f"ablation_{tag}_d{ds_num}"


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ERPDiff ablation runner for dataset1, dataset2, and dataset3."
    )
    p.add_argument(
        "--datasets", nargs="+", choices=DATASET_CHOICES, default=DATASET_CHOICES,
        help="Which datasets to run. Default: dataset1, dataset2, and dataset3.",
    )
    p.add_argument(
        "--experiments", nargs="+", choices=EXPERIMENT_NAMES, default=EXPERIMENT_NAMES,
        metavar="EXP",
        help=f"Which experiments to run. Choices: {EXPERIMENT_NAMES}. Default: all.",
    )
    p.add_argument("--epochs",  type=int, default=300,  help="Training epochs per run.")
    p.add_argument("--runs",    type=int, default=5,    help="Number of repeated runs (seeds).")
    p.add_argument("--seed",    type=int, default=42,   help="Base random seed.")
    p.add_argument("--device",  type=str, default=None, help="Torch device, e.g. cuda:0.")
    p.add_argument(
        "--spec-file", type=str, default=None,
        help="Optional experiment spec passed through to erpdiff_train.py for consistent hyperparameters.",
    )
    p.add_argument(
        "--out-root", type=str, default=None,
        help="Root output directory. Default: ./ablation_results_<date>/ next to this script.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip any experiment whose output dir already contains finetune_report.json.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print all commands that would be run, but don't actually execute them.",
    )
    p.add_argument(
        "--lambda-intra", type=float, default=0.05,
        help="Intra-branch consistency loss weight (passed through to all experiments).",
    )
    p.add_argument(
        "--warmup-epochs", type=int, default=10,
        help="Warmup epochs for full / non-exp8 experiments.",
    )
    p.add_argument(
        "--early-stop-patience", type=int, default=35,
        help="Finetune early-stopping patience.",
    )
    return p.parse_args()


# ============================================================
# Utilities
# ============================================================

def find_train_script(script_dir: Path) -> Path:
    """
    Prefer erpdiff_train_modified.py (has --clb-bsd-beta support).
    Fall back to erpdiff_train.py and warn if exp7 is in the run list.
    """
    modified = script_dir / "erpdiff_train_modified.py"
    original = script_dir / "erpdiff_train.py"
    if modified.exists():
        return modified
    if original.exists():
        return original
    raise FileNotFoundError(
        f"Neither erpdiff_train_modified.py nor erpdiff_train.py found in {script_dir}"
    )


def already_done(out_dir: Path) -> bool:
    """Check if a previous run completed successfully (finetune_report.json exists)."""
    return (out_dir / "finetune_report.json").exists()


def _fmt_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def build_command(
    train_script: Path,
    dataset: str,
    exp_extra_args: List[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        sys.executable, str(train_script),
        "--dataset",  dataset,
        "--mode",     "both",
        "--epochs",   str(args.epochs),
        "--runs",     str(args.runs),
        "--seed",     str(args.seed),
        "--out-dir",  str(out_dir),
        "--lambda-intra", str(args.lambda_intra),
        "--finetune-early-stop-patience", str(args.early_stop_patience),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.spec_file:
        cmd += ["--spec-file", args.spec_file]

    # Warmup: exp8 overrides with 0; all others use the configured value
    if "--warmup-epochs" not in exp_extra_args:
        cmd += ["--warmup-epochs", str(args.warmup_epochs)]

    cmd += exp_extra_args
    return cmd


# ============================================================
# Main runner
# ============================================================

def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).parent.resolve()
    train_script = find_train_script(script_dir)

    # Warn if exp7 is requested but the modified script is not available
    if False and "exp7_no_bsd" in args.experiments:
        print(
            "\n[WARNING] erpdiff_train_modified.py not found — using erpdiff_train.py.\n"
            "  Exp7 (--clb-bsd-beta 0.0) requires the modified file.\n"
            "  Copy erpdiff_train_modified.py and erpdiff_config_modified.py to the project\n"
            "  directory (replacing the originals) before running exp7, or exp7 will be\n"
            "  silently skipped.\n"
        )

    # Output root
    date_tag = datetime.now().strftime("%y%m%d_%H%M")
    out_root = Path(args.out_root) if args.out_root else script_dir / f"ablation_results_{date_tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve the experiment configs to run
    exp_map: Dict[str, Tuple[str, List[str]]] = {
        name: (desc, extra) for name, desc, extra in EXPERIMENT_CONFIGS
    }
    selected_experiments = [
        (name, exp_map[name][0], exp_map[name][1])
        for name in args.experiments
    ]

    # Summary printout
    total_jobs = len(args.datasets) * len(selected_experiments)
    print("=" * 70)
    print("ERPDiff Ablation Runner")
    print("=" * 70)
    print(f"  Train script : {train_script.name}")
    print(f"  Datasets     : {args.datasets}")
    print(f"  Experiments  : {[n for n, _, _ in selected_experiments]}")
    print(f"  Epochs       : {args.epochs}  |  Runs : {args.runs}  |  Seed : {args.seed}")
    print(f"  Spec file    : {args.spec_file}")
    print(f"  Output root  : {out_root}")
    print(f"  Resume mode  : {args.resume}")
    print(f"  Dry run      : {args.dry_run}")
    print(f"  Total jobs   : {total_jobs}")
    print("=" * 70)

    # Execution log
    log_path = out_root / "ablation_run_log.json"
    run_log: List[dict] = []

    job_idx = 0
    global_start = time.perf_counter()

    for dataset in args.datasets:
        for exp_name, exp_desc, exp_extra in selected_experiments:
            job_idx += 1
            exp_out_dir = out_root / exp_out_name(exp_name, dataset)
            exp_out_dir.mkdir(parents=True, exist_ok=True)

            # # Skip exp7 if modified script not available
            # if exp_name == "exp7_no_bsd" and not using_modified:
            #     print(f"\n[{job_idx}/{total_jobs}] SKIP {exp_name} / {dataset} "
            #           f"(erpdiff_train_modified.py required)")
            #     run_log.append({
            #         "job": job_idx, "dataset": dataset, "experiment": exp_name,
            #         "status": "skipped_no_modified_script", "out_dir": str(exp_out_dir),
            #     })
            #     continue

            # Resume check
            if args.resume and already_done(exp_out_dir):
                print(f"\n[{job_idx}/{total_jobs}] SKIP {exp_name} / {dataset} "
                      f"(finetune_report.json exists, --resume)")
                run_log.append({
                    "job": job_idx, "dataset": dataset, "experiment": exp_name,
                    "status": "skipped_resume", "out_dir": str(exp_out_dir),
                })
                continue

            cmd = build_command(train_script, dataset, exp_extra, exp_out_dir, args)

            print(f"\n{'─'*70}")
            print(f"[{job_idx}/{total_jobs}]  {exp_desc}")
            print(f"  Dataset : {dataset}")
            print(f"  Out dir : {exp_out_dir}")
            print(f"  Command : {' '.join(cmd)}")

            if args.dry_run:
                run_log.append({
                    "job": job_idx, "dataset": dataset, "experiment": exp_name,
                    "status": "dry_run", "cmd": cmd, "out_dir": str(exp_out_dir),
                })
                continue

            # Save per-job metadata before running
            meta_path = exp_out_dir / "ablation_meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "job": job_idx,
                    "dataset": dataset,
                    "experiment": exp_name,
                    "description": exp_desc,
                    "extra_args": exp_extra,
                    "command": cmd,
                    "started_at": datetime.now().isoformat(),
                }, f, indent=2)

            t0 = time.perf_counter()
            result = subprocess.run(cmd, cwd=str(script_dir))
            elapsed = time.perf_counter() - t0

            status = "success" if result.returncode == 0 else f"failed_rc{result.returncode}"
            print(f"\n  → {status}  (elapsed: {_fmt_elapsed(elapsed)})")

            # Update per-job metadata with result
            with open(meta_path, "r+", encoding="utf-8") as f:
                meta = json.load(f)
            meta["finished_at"]   = datetime.now().isoformat()
            meta["elapsed_s"]     = round(elapsed, 1)
            meta["returncode"]    = result.returncode
            meta["status"]        = status
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            run_log.append({
                "job": job_idx, "dataset": dataset, "experiment": exp_name,
                "status": status, "elapsed_s": round(elapsed, 1),
                "out_dir": str(exp_out_dir),
            })

            # Persist run log after every job (so partial runs are recoverable)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(run_log, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - global_start
    n_success  = sum(1 for r in run_log if r["status"] == "success")
    n_failed   = sum(1 for r in run_log if r["status"].startswith("failed"))
    n_skipped  = sum(1 for r in run_log if r["status"].startswith("skipped"))
    n_dry      = sum(1 for r in run_log if r["status"] == "dry_run")

    print(f"\n{'='*70}")
    print("Ablation run complete")
    print(f"  Total elapsed : {_fmt_elapsed(total_elapsed)}")
    print(f"  Success       : {n_success}")
    print(f"  Failed        : {n_failed}")
    print(f"  Skipped       : {n_skipped}")
    if n_dry:
        print(f"  Dry-run       : {n_dry}")
    print(f"  Log           : {log_path}")
    print("=" * 70)

    if n_failed:
        print("\n[FAILED JOBS]")
        for r in run_log:
            if r["status"].startswith("failed"):
                print(f"  {r['dataset']} / {r['experiment']}  →  {r['out_dir']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
