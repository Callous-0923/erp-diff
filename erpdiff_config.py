"""
ERPDiff — Configuration and hyperparameters.
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

try:
    import yaml
except ImportError:
    yaml = None

# =========================
# Dataset paths
# =========================
_DEFAULT_DATASET_DIRS: Dict[str, str] = {
    "dataset1": r"D:\files\datasets\erp-dataset1_preprocessed", # mocnn dataset1
    "dataset2": r"D:\files\codes\EEGInception\PreData_MOCNN",
    "dataset3": r"D:\files\datasets\BCI_Comp_III_Wads_2004_preprocessed",
}


def _build_dataset_dirs() -> Dict[str, str]:
    """Build dataset dirs, allowing env var overrides for cloud deployment.
    Set ERPDIFF_DATASET1_DIR / _DATASET2_DIR / _DATASET3_DIR to override."""
    d = dict(_DEFAULT_DATASET_DIRS)
    for key in ("dataset1", "dataset2", "dataset3"):
        env_val = os.environ.get(f"ERPDIFF_{key.upper()}_DIR")
        if env_val:
            d[key] = env_val
    return d


DATASET_DIRS: Dict[str, str] = _build_dataset_dirs()

PRE_DATA_DIR: str = DATASET_DIRS["dataset3"]
OUT_BASE_DIR: str = os.path.dirname(__file__)
TEMPORAL_BIAS_SIGMA_MODES: Tuple[str, ...] = ("shared", "per_head")


def resolve_out_dir(dataset: str, base_dir: str = OUT_BASE_DIR) -> str:
    """Auto-generate output directory: erpdiff_output_YYMMDD_dN_K"""
    from datetime import datetime
    today  = datetime.now().strftime("%y%m%d")
    ds_tag = f"d{dataset[-1]}"
    prefix = f"erpdiff_output_{today}_{ds_tag}_"

    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)
    ] if os.path.isdir(base_dir) else []

    max_idx = 0
    for d in existing:
        suffix = d[len(prefix):]
        try:
            max_idx = max(max_idx, int(suffix))
        except ValueError:
            pass

    return os.path.join(base_dir, f"{prefix}{max_idx + 1}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    # --- Core training ---
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset: str = "dataset3"
    pretrain_batch_size: int = 256
    finetune_batch_size: int = 128
    pretrain_lr: float = 0.002
    finetune_lr: float = 0.0012
    num_workers: int = 4
    epochs: int = 10
    min_lr_ratio: float = 0.1
    optimizer: str = "adam"
    dropout_p: float = 0.25
    focal_alpha: float = 0.80                  # Focal loss alpha (positive/minority class weight, >0.5)
    focal_gamma: float = 2.0
    aux_branch_ce_weight: float = 0.2
    train_ratio: float = 0.8
    val_ratio_in_train: float = 0.2
    eps: float = 1e-8
    pretrain_weight_decay: float = 0.0005
    finetune_weight_decay: float = 0.0005

    # --- ERPDiff ablation switches ---
    use_temporal_bias: bool = True       # Dual-sigma Gaussian temporal bias
    temporal_bias_sigma_mode: str = "shared"  # shared scalar sigma vs per-head sigma
    use_alpha_gate: bool = True          # Alpha-gated cross-attention
    min_gate: float = 0.1               # Floor for alpha gate
    lambda_comp: float = 0.1            # Complementarity loss weight (0 = disabled)
    comp_margin: float = 0.5            # Margin for soft decorrelation
    clb_bsd_beta: float = 0.5          # BSD self-distillation weight for CLB pretrain (0 = CE-only, Ablation Exp7)


# =========================
# YAML parsing helpers
# =========================

def _parse_min_lr_ratio(raw_min_lr) -> float:
    if isinstance(raw_min_lr, (int, float)):
        return float(raw_min_lr)
    if isinstance(raw_min_lr, str):
        tokens = raw_min_lr.replace(" ", "").replace("lr", "").split("*")
        try:
            return float(tokens[0]) if tokens else 0.1
        except Exception:
            return 0.1
    return 0.1


def _to_float(raw, default: float) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except Exception:
            return default
    return default


def _to_int(raw, default: int) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(float(raw.strip()))
        except Exception:
            return default
    return default


def _parse_temporal_bias_sigma_mode(raw_mode, default: str) -> str:
    if not isinstance(raw_mode, str):
        return default
    mode = raw_mode.strip().lower()
    return mode if mode in TEMPORAL_BIAS_SIGMA_MODES else default


def load_spec_to_config(spec_path: str, cfg: TrainConfig) -> Tuple[TrainConfig, int]:
    runs = 1
    if not os.path.exists(spec_path) or yaml is None:
        return cfg, runs
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return cfg, runs

    spec  = data.get("ERPDIFF_EXPERIMENT_SPEC", data.get("ITSEF_EXPERIMENT_SPEC", {}))
    table = spec.get("table2_hyperparams", {})
    pre   = table.get("pretrain", {})
    fin   = table.get("finetune", {})

    cfg.pretrain_batch_size = _to_int(pre.get("batch_size",   cfg.pretrain_batch_size), cfg.pretrain_batch_size)
    cfg.pretrain_lr         = _to_float(pre.get("initial_lr", cfg.pretrain_lr),         cfg.pretrain_lr)
    cfg.finetune_batch_size = _to_int(fin.get("batch_size",   cfg.finetune_batch_size), cfg.finetune_batch_size)
    cfg.finetune_lr         = _to_float(fin.get("initial_lr", cfg.finetune_lr),         cfg.finetune_lr)
    cfg.num_workers         = _to_int(table.get("num_workers", cfg.num_workers),         cfg.num_workers)
    cfg.epochs              = _to_int(table.get("epochs",      cfg.epochs),              cfg.epochs)

    opt_sched = spec.get("optimizer_and_scheduler", {})
    cfg.min_lr_ratio = _parse_min_lr_ratio(opt_sched.get("min_lr", cfg.min_lr_ratio))
    weight_decay = opt_sched.get("weight_decay", None)
    if weight_decay is not None:
        wd = _to_float(weight_decay, cfg.pretrain_weight_decay)
        cfg.pretrain_weight_decay = wd
        cfg.finetune_weight_decay = wd
    cfg.pretrain_weight_decay = _to_float(pre.get("weight_decay", cfg.pretrain_weight_decay), cfg.pretrain_weight_decay)
    cfg.finetune_weight_decay = _to_float(fin.get("weight_decay", cfg.finetune_weight_decay), cfg.finetune_weight_decay)

    split = spec.get("data_split", {})
    cfg.train_ratio        = 0.8 if split.get("train_test_ratio", "8:2") == "8:2" else cfg.train_ratio
    cfg.val_ratio_in_train = split.get("validation_from_train", cfg.val_ratio_in_train)

    repro = spec.get("reproducibility", {})
    runs  = repro.get("runs_per_method", runs)

    model_cfg = spec.get("model", {}) if isinstance(spec, dict) else {}
    if isinstance(model_cfg, dict) and "dropout_p" in model_cfg:
        try:
            cfg.dropout_p = float(model_cfg["dropout_p"])
        except Exception:
            pass

    loss_cfg = spec.get("loss", {}) if isinstance(spec, dict) else {}
    if isinstance(loss_cfg, dict):
        focal_cfg = loss_cfg.get("focal", {})
        if isinstance(focal_cfg, dict):
            if "alpha" in focal_cfg:
                try: cfg.focal_alpha = float(focal_cfg["alpha"])
                except Exception: pass
            if "gamma" in focal_cfg:
                try: cfg.focal_gamma = float(focal_cfg["gamma"])
                except Exception: pass

    if "dataset" in spec:
        ds = str(spec["dataset"]).strip()
        if ds in ("dataset1", "dataset2", "dataset3"):
            cfg.dataset = ds

    # --- ERPDiff ablation switches from yaml ---
    ablation = spec.get("ablation", {})
    if isinstance(ablation, dict):
        if "use_temporal_bias" in ablation:
            cfg.use_temporal_bias = bool(ablation["use_temporal_bias"])
        if "temporal_bias_sigma_mode" in ablation:
            cfg.temporal_bias_sigma_mode = _parse_temporal_bias_sigma_mode(
                ablation["temporal_bias_sigma_mode"], cfg.temporal_bias_sigma_mode
            )
        if "use_alpha_gate" in ablation:
            cfg.use_alpha_gate = bool(ablation["use_alpha_gate"])
        if "min_gate" in ablation:
            cfg.min_gate = _to_float(ablation["min_gate"], cfg.min_gate)
        if "lambda_comp" in ablation:
            cfg.lambda_comp = _to_float(ablation["lambda_comp"], cfg.lambda_comp)
        if "comp_margin" in ablation:
            cfg.comp_margin = _to_float(ablation["comp_margin"], cfg.comp_margin)

    return cfg, runs
