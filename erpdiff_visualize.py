"""
ERPDiff — Visualization & Interpretability Analysis.

Generates all figures needed for the ERPDiff paper:

  Fig 1. Training curves (loss & accuracy) for pretrain and finetune stages
  Fig 2. λ / sigma_signal / sigma_noise evolution over training epochs
  Fig 3. Attention heatmaps: attn1 (signal head) vs attn2 (noise head) vs differential
  Fig 4. Gaussian temporal bias profiles (learned sigma_signal vs sigma_noise)
  Fig 5. Confusion matrices per subject
  Fig 6. Per-subject accuracy comparison bar chart (ERPDiff vs baselines)
  Fig 7. Ablation study bar chart

Usage:
    # After training, point to the output directory:
    python erpdiff_visualize.py --run-dir erpdiff_output_260316_d3_1

    # Generate attention heatmaps from a saved model:
    python erpdiff_visualize.py --run-dir ... --mode attn --subject subject_A_Train

    # All figures at once:
    python erpdiff_visualize.py --run-dir ... --mode all
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# Utility: load JSON logs
# ============================================================

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_subject_dirs(run_dir: str) -> List[str]:
    """Find all subject subdirectories that contain finetune_log.json."""
    dirs = []
    for name in sorted(os.listdir(run_dir)):
        sub_dir = os.path.join(run_dir, name)
        if os.path.isdir(sub_dir) and os.path.exists(os.path.join(sub_dir, "finetune_log.json")):
            dirs.append(sub_dir)
    return dirs


def _coerce_sigma_values(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _plot_sigma_history(ax, epochs: List[int], sigma_history: List[Any], label_prefix: str) -> None:
    valid = []
    for epoch, sigma_value in zip(epochs, sigma_history):
        sigma_values = _coerce_sigma_values(sigma_value)
        if sigma_values is not None:
            valid.append((epoch, sigma_values))
    if not valid:
        return

    plot_epochs = [epoch for epoch, _ in valid]
    sigma_matrix = np.asarray([values for _, values in valid], dtype=float)
    if sigma_matrix.shape[1] == 1:
        ax.plot(plot_epochs, sigma_matrix[:, 0], label=label_prefix, alpha=0.8)
        return

    ax.plot(plot_epochs, sigma_matrix.mean(axis=1), label=f"{label_prefix}-mean", alpha=0.9, linewidth=2.0)
    for head_idx in range(sigma_matrix.shape[1]):
        ax.plot(
            plot_epochs,
            sigma_matrix[:, head_idx],
            label=f"{label_prefix}-h{head_idx}",
            alpha=0.45,
            linestyle="--",
        )


# ============================================================
# Fig 1: Training curves
# ============================================================

def plot_training_curves(run_dir: str, save_dir: str):
    """Plot pretrain and finetune loss/accuracy curves."""
    pretrain_log = os.path.join(run_dir, "pretrain_log.json")
    if not os.path.exists(pretrain_log):
        print("[Skip] No pretrain_log.json found.")
        return

    data = load_json(pretrain_log)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ERPDiff Pretrain Training Curves", fontsize=14, fontweight="bold")

    # CLB-BSD history
    if "clb_history" in data:
        clb = data["clb_history"]
        epochs = [r["epoch"] for r in clb]
        axes[0, 0].plot(epochs, [r["train_loss"] for r in clb], label="Train Loss")
        axes[0, 0].plot(epochs, [r["val_loss"] for r in clb], label="Val Loss")
        axes[0, 0].set_title("CLB Branch (BSD) — Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, [r["train_acc"] for r in clb], label="Train Acc")
        axes[0, 1].plot(epochs, [r["val_acc"] for r in clb], label="Val Acc")
        axes[0, 1].set_title("CLB Branch (BSD) — Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # RBB history
    if "rbb_history" in data:
        rbb = data["rbb_history"]
        epochs = [r["epoch"] for r in rbb]
        axes[1, 0].plot(epochs, [r["train_loss"] for r in rbb], label="Train Loss")
        axes[1, 0].plot(epochs, [r["val_loss"] for r in rbb], label="Val Loss")
        axes[1, 0].set_title("RBB Branch (Focal) — Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, [r["train_acc"] for r in rbb], label="Train Acc")
        axes[1, 1].plot(epochs, [r["val_acc"] for r in rbb], label="Val Acc")
        axes[1, 1].set_title("RBB Branch (Focal) — Accuracy")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig1_pretrain_curves.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")

    # Finetune curves per subject
    subject_dirs = find_subject_dirs(run_dir)
    if not subject_dirs:
        return

    n_subs = len(subject_dirs)
    fig, axes = plt.subplots(n_subs, 2, figsize=(14, 4 * n_subs), squeeze=False)
    fig.suptitle("ERPDiff Finetune Training Curves", fontsize=14, fontweight="bold")

    for i, sd in enumerate(subject_dirs):
        sid = os.path.basename(sd)
        log = load_json(os.path.join(sd, "finetune_log.json"))
        hist = log["history"]
        epochs = [r["epoch"] for r in hist]

        axes[i, 0].plot(epochs, [r["train_loss"] for r in hist], label="Train")
        axes[i, 0].plot(epochs, [r["val_loss"] for r in hist], label="Val")
        axes[i, 0].set_title(f"{sid} — Loss")
        axes[i, 0].set_xlabel("Epoch")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(epochs, [r["train_acc"] for r in hist], label="Train")
        axes[i, 1].plot(epochs, [r["val_acc"] for r in hist], label="Val")
        axes[i, 1].set_title(f"{sid} — Accuracy")
        axes[i, 1].set_xlabel("Epoch")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig1_finetune_curves.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ============================================================
# Fig 2: λ and sigma evolution over epochs
# ============================================================

def plot_lambda_sigma_evolution(run_dir: str, save_dir: str):
    """Plot how λ, sigma_signal, sigma_noise evolve during training."""
    subject_dirs = find_subject_dirs(run_dir)
    if not subject_dirs:
        print("[Skip] No subject finetune logs found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("ERPDiff — Learned Parameter Evolution During Fine-tuning", fontsize=14, fontweight="bold")

    for sd in subject_dirs:
        sid = os.path.basename(sd)
        hist = load_json(os.path.join(sd, "finetune_log.json"))["history"]
        epochs = [r["epoch"] for r in hist]

        # λ (RBB self-attention)
        lambdas = [r.get("rbb_self_attn_lambda") for r in hist]
        if any(v is not None for v in lambdas):
            axes[0].plot(epochs, lambdas, label=sid, alpha=0.8)

        # sigma_signal
        sig_s = [r.get("sigma_signal") for r in hist]
        _plot_sigma_history(axes[1], epochs, sig_s, sid)

        # sigma_noise
        sig_n = [r.get("sigma_noise") for r in hist]
        _plot_sigma_history(axes[2], epochs, sig_n, sid)

    axes[0].set_title("λ (Differential Attention)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("λ value")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("σ_signal (Narrow Gaussian)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("σ value")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("σ_noise (Wide Gaussian)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("σ value")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig2_lambda_sigma_evolution.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ============================================================
# Fig 3: Attention heatmaps (requires model + data)
# ============================================================

def plot_attention_heatmaps(
    model_path: str,
    data_dir: str,
    subject_id: str,
    save_dir: str,
    cfg_dict: Optional[dict] = None,
    n_samples: int = 8,
):
    """
    Extract and visualize attention maps from a trained ERPDiff model.
    Shows attn1 (signal), attn2 (noise), and attn1 - λ·attn2 (differential).
    """
    if not HAS_TORCH:
        print("[Skip] PyTorch not available for attention extraction.")
        return

    from erpdiff_config import TrainConfig
    from erpdiff_data import build_subject_splits
    from erpdiff_model import ERPDiff

    cfg = TrainConfig()
    if cfg_dict:
        for k, v in cfg_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # Find subject pkl
    import glob
    pkls = glob.glob(os.path.join(data_dir, f"{subject_id}*.pkl"))
    if not pkls:
        print(f"[Skip] No pkl found for {subject_id} in {data_dir}")
        return

    _, _, test_ds = build_subject_splits(subject_id, pkls[0], cfg)
    if len(test_ds) == 0:
        print(f"[Skip] Empty test set for {subject_id}")
        return

    device = "cpu"
    state = torch.load(model_path, map_location=device)

    # Infer C, T from test dataset
    C, T = test_ds.C, test_ds.T
    model = ERPDiff(
        in_channels=C,
        n_samples=T,
        use_temporal_bias=cfg.use_temporal_bias,
        temporal_bias_sigma_mode=getattr(cfg, "temporal_bias_sigma_mode", "shared"),
        use_alpha_gate=cfg.use_alpha_gate,
        min_gate=cfg.min_gate,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Collect attention maps for a few target and nontarget samples
    target_maps = {"attn1": [], "attn2": [], "labels": []}
    count = 0
    for i in range(min(len(test_ds), 200)):
        x_t, y_t, _ = test_ds[i]
        if count >= n_samples:
            break
        x_in = x_t.unsqueeze(0).to(device)
        _ = model(x_in, alpha=0.5)
        attn = model.rbb_temporal_attn
        if attn.last_attn1 is not None:
            target_maps["attn1"].append(attn.last_attn1[0].cpu().numpy())  # [H, T, T]
            target_maps["attn2"].append(attn.last_attn2[0].cpu().numpy())
            target_maps["labels"].append(int(y_t.item()))
            count += 1

    if not target_maps["attn1"]:
        print("[Skip] No attention maps collected.")
        return

    # Plot: average across collected samples, head 0
    avg_attn1 = np.mean([m[0] for m in target_maps["attn1"]], axis=0)  # [T, T]
    avg_attn2 = np.mean([m[0] for m in target_maps["attn2"]], axis=0)

    lam = float(attn.last_lambda.item()) if attn.last_lambda is not None else 0.8
    diff_attn = avg_attn1 - lam * avg_attn2

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"ERPDiff Attention Heatmaps — {subject_id} (head 0, avg over {count} samples)",
                 fontsize=13, fontweight="bold")

    im0 = axes[0].imshow(avg_attn1, aspect="auto", cmap="YlOrRd")
    axes[0].set_title(f"attn₁ (Signal Head, σ={cfg.use_temporal_bias})")
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(avg_attn2, aspect="auto", cmap="YlOrRd")
    axes[1].set_title("attn₂ (Noise Head)")
    axes[1].set_xlabel("Key position")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff_attn, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    axes[2].set_title(f"attn₁ − λ·attn₂  (λ={lam:.3f})")
    axes[2].set_xlabel("Key position")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    path = os.path.join(save_dir, f"fig3_attn_heatmap_{subject_id}.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ============================================================
# Fig 4: Gaussian temporal bias profiles
# ============================================================

def plot_gaussian_bias_profiles(run_dir: str, save_dir: str):
    """Visualize learned Gaussian bias shapes from final sigma values."""
    subject_dirs = find_subject_dirs(run_dir)
    if not subject_dirs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ERPDiff — Learned Gaussian Temporal Bias Profiles (Final Epoch)",
                 fontsize=13, fontweight="bold")

    for sd in subject_dirs:
        sid = os.path.basename(sd)
        hist = load_json(os.path.join(sd, "finetune_log.json"))["history"]
        # Get final epoch values
        final = hist[-1] if hist else {}
        sig_s = _coerce_sigma_values(final.get("sigma_signal"))
        sig_n = _coerce_sigma_values(final.get("sigma_noise"))
        if sig_s is None or sig_n is None:
            continue

        t = np.arange(-20, 21)
        if len(sig_s) != 1 or len(sig_n) != 1:
            for head_idx, sigma_value in enumerate(sig_s):
                bias_signal = np.exp(-0.5 * (t / max(sigma_value, 1.0)) ** 2)
                axes[0].plot(t, bias_signal, label=f"{sid}-h{head_idx} (蟽={sigma_value:.1f})", alpha=0.45)
            for head_idx, sigma_value in enumerate(sig_n):
                bias_noise = np.exp(-0.5 * (t / max(sigma_value, 1.0)) ** 2)
                axes[1].plot(t, bias_noise, label=f"{sid}-h{head_idx} (蟽={sigma_value:.1f})", alpha=0.45)
            continue

        sig_s = sig_s[0]
        sig_n = sig_n[0]
        bias_signal = np.exp(-0.5 * (t / max(sig_s, 1.0)) ** 2)
        bias_noise  = np.exp(-0.5 * (t / max(sig_n, 1.0)) ** 2)

        axes[0].plot(t, bias_signal, label=f"{sid} (σ={sig_s:.1f})", alpha=0.8)
        axes[1].plot(t, bias_noise,  label=f"{sid} (σ={sig_n:.1f})", alpha=0.8)

    axes[0].set_title("Signal Head Bias (Narrow)")
    axes[0].set_xlabel("Relative token distance")
    axes[0].set_ylabel("Bias weight (exp scale)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0, color="gray", linestyle="--", alpha=0.5)

    axes[1].set_title("Noise Head Bias (Wide)")
    axes[1].set_xlabel("Relative token distance")
    axes[1].set_ylabel("Bias weight (exp scale)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig4_gaussian_bias_profiles.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ============================================================
# Fig 5: Confusion matrices
# ============================================================

def plot_confusion_matrices(run_dir: str, save_dir: str):
    """
    Plot confusion matrices per subject.
    Requires running evaluation to collect predictions — this function
    reads from finetune_log.json if confusion data is available, or
    generates placeholder layout that you can fill after evaluation.
    """
    subject_dirs = find_subject_dirs(run_dir)
    if not subject_dirs:
        return

    report_path = os.path.join(run_dir, "finetune_report.json")
    if not os.path.exists(report_path):
        print("[Skip] No finetune_report.json found.")
        return

    report = load_json(report_path)
    n_subs = len(subject_dirs)
    cols = min(n_subs, 4)
    rows = (n_subs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    fig.suptitle("ERPDiff — Per-Subject Test Accuracy", fontsize=14, fontweight="bold")

    for i, sd in enumerate(subject_dirs):
        r, c = divmod(i, cols)
        sid = os.path.basename(sd)
        acc = report.get(sid, {}).get("test_acc", 0)
        axes[r][c].bar(["Accuracy"], [acc * 100], color="#4A90D9", width=0.4)
        axes[r][c].set_ylim(0, 100)
        axes[r][c].set_title(sid, fontsize=10)
        axes[r][c].set_ylabel("%")
        for spine in axes[r][c].spines.values():
            spine.set_visible(False)

    # Hide unused axes
    for i in range(n_subs, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig5_per_subject_accuracy.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ============================================================
# Fig 6: Comparison bar chart (ERPDiff vs baselines)
# ============================================================

def plot_comparison_bar(
    erpdiff_results: Dict[str, float],
    baseline_results: Dict[str, Dict[str, float]],
    save_dir: str,
    metric_name: str = "Acc",
):
    """
    Bar chart comparing ERPDiff against baseline methods.

    Args:
        erpdiff_results:   {subject_id: accuracy, ...}
        baseline_results:  {method_name: {subject_id: accuracy, ...}, ...}
        save_dir:          Output directory.
        metric_name:       Metric label for y-axis.
    """
    methods = list(baseline_results.keys()) + ["ERPDiff"]
    subjects = sorted(erpdiff_results.keys())
    n_methods = len(methods)
    n_subjects = len(subjects)

    x = np.arange(n_subjects + 1)  # +1 for average
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(12, 2 * n_subjects), 6))

    for j, method in enumerate(methods):
        if method == "ERPDiff":
            vals = [erpdiff_results.get(s, 0) * 100 for s in subjects]
        else:
            vals = [baseline_results[method].get(s, 0) * 100 for s in subjects]
        avg = np.mean(vals)
        vals_with_avg = vals + [avg]
        bars = ax.bar(x + j * width, vals_with_avg, width, label=method, alpha=0.85)

    ax.set_xlabel("Subject")
    ax.set_ylabel(f"{metric_name} (%)")
    ax.set_title(f"ERPDiff vs Baselines — {metric_name}", fontweight="bold")
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(subjects + ["Avg"], rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig6_comparison_bar.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ============================================================
# Fig 7: Ablation study bar chart
# ============================================================

def plot_ablation_bar(
    ablation_results: Dict[str, float],
    save_dir: str,
):
    """
    Bar chart for ablation study.

    Args:
        ablation_results:  {variant_name: avg_accuracy, ...}
            e.g. {"ERPDiff (full)": 0.92, "w/o temporal bias": 0.89, ...}
    """
    variants = list(ablation_results.keys())
    accs = [ablation_results[v] * 100 for v in variants]

    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 1.5), 5))
    colors = ["#4A90D9"] * len(variants)
    colors[0] = "#E74C3C"  # highlight full model

    bars = ax.bar(variants, accs, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("ERPDiff — Ablation Study", fontweight="bold")
    ax.set_ylim(min(accs) - 3, max(accs) + 3)
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig7_ablation.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ERPDiff visualization & interpretability analysis.")
    parser.add_argument("--run-dir", required=True, help="Path to training output directory.")
    parser.add_argument("--save-dir", default=None, help="Where to save figures (default: run-dir/figures).")
    parser.add_argument("--mode", default="all",
                        choices=["all", "curves", "lambda", "attn", "bias", "subjects", "comparison", "ablation"],
                        help="Which figure(s) to generate.")
    parser.add_argument("--subject", default=None, help="Subject ID for attention heatmaps.")
    parser.add_argument("--data-dir", default=None, help="Preprocessed data dir (for attention extraction).")
    args = parser.parse_args()

    if not HAS_MPL:
        print("[Error] matplotlib is required. Install with: pip install matplotlib")
        return

    save_dir = args.save_dir or os.path.join(args.run_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)

    if args.mode in ("all", "curves"):
        plot_training_curves(args.run_dir, save_dir)

    if args.mode in ("all", "lambda"):
        plot_lambda_sigma_evolution(args.run_dir, save_dir)

    if args.mode in ("all", "bias"):
        plot_gaussian_bias_profiles(args.run_dir, save_dir)

    if args.mode in ("all", "subjects"):
        plot_confusion_matrices(args.run_dir, save_dir)

    if args.mode == "attn":
        if not args.subject or not args.data_dir:
            print("[Error] --subject and --data-dir required for attention mode.")
            return
        model_path = os.path.join(args.run_dir, args.subject, "finetune_erpdiff.pth")
        if not os.path.exists(model_path):
            print(f"[Error] Model not found: {model_path}")
            return
        plot_attention_heatmaps(model_path, args.data_dir, args.subject, save_dir)

    print(f"\n[Done] All figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
