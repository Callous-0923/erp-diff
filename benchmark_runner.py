"""
ERPDiff vs. Baselines 鈥?Unified Benchmark Runner.

Runs all models on all datasets with both B1 and B2-Full protocols,
plus ERPDiff's own pretrain+finetune pipeline.

Usage:
    python benchmark_runner.py
    python benchmark_runner.py --datasets dataset2 --models eegnet icnn --protocols B1
    python benchmark_runner.py --seeds 42 --datasets dataset3 --skip-erpdiff
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Ensure project root is on path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from erpdiff_config import DATASET_DIRS, TrainConfig, set_global_seed
from erpdiff_data import (
    EEGTrialsDataset,
    build_generic_splits_with_meta,
    build_subject_splits,
    discover_subject_pkls,
    make_loader,
)
from benchmark_baseline_trainer import (
    ALL_BASELINE_MODELS,
    evaluate_baseline_all_protocols,
    get_baseline_model,
    train_b1,
    train_b2,
)
from benchmark_utils import (
    collect_probs_and_labels,
    collect_probs_with_ensemble,
    collect_probs_with_tta,
    count_parameters,
)
from benchmark_char_metrics import (
    compute_char_acc_dataset1_rsvp,
    compute_char_acc_dataset2,
    compute_char_acc_dataset3,
)


# ============================================================
# ERPDiff wrapper
# ============================================================

def run_erpdiff_for_subject(
    subject_id: str,
    pkl_path: str,
    cfg: TrainConfig,
    out_dir: str,
    pretrained_clb: str,
    pretrained_rbb: str,
    warmup_epochs: int = 10,
    lambda_intra: float = 0.05,
    early_stop_patience: int = 35,
    early_stop_min_delta: float = 1e-4,
) -> Dict[str, Any]:
    """
    Run ERPDiff finetune for a single subject and return test metrics.
    This calls the existing finetune logic from erpdiff_train.py.
    """
    from erpdiff_train import finetune_stage

    finetune_stage(
        predata_dir=os.path.dirname(pkl_path),
        out_dir=out_dir,
        cfg=cfg,
        pretrained_clb=pretrained_clb,
        pretrained_rbb=pretrained_rbb,
        subjects=[subject_id],
        warmup_epochs=warmup_epochs,
        lambda_intra=lambda_intra,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )

    # Read the finetune log to get test metrics
    log_path = os.path.join(out_dir, subject_id, "finetune_log.json")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        return log.get("test_metrics", {})
    return {}


def run_erpdiff_full(
    dataset_name: str,
    predata_dir: str,
    seed: int,
    out_dir: str,
    cfg: TrainConfig,
    pretrain_patience: int = 14,
    finetune_patience: int = 35,
    warmup_epochs: int = 10,
    lambda_intra: float = 0.05,
) -> Dict[str, Any]:
    """
    Run ERPDiff pretrain + finetune for all subjects in a dataset.
    Returns per-subject metrics.
    """
    from erpdiff_train import pretrain_stage, finetune_stage

    erpdiff_dir = os.path.join(out_dir, "erpdiff")
    os.makedirs(erpdiff_dir, exist_ok=True)

    t0 = time.time()

    # Pretrain
    print(f"\n{'='*60}")
    print(f"[ERPDiff] Pretrain on {dataset_name} (seed={seed})")
    print(f"{'='*60}")
    pretrained = pretrain_stage(
        predata_dir=predata_dir,
        out_dir=erpdiff_dir,
        cfg=cfg,
        early_stop_patience=pretrain_patience,
        early_stop_min_delta=1e-4,
    )
    pretrain_time = time.time() - t0

    # Finetune
    print(f"\n{'='*60}")
    print(f"[ERPDiff] Finetune on {dataset_name} (seed={seed})")
    print(f"{'='*60}")
    t1 = time.time()
    finetune_stage(
        predata_dir=predata_dir,
        out_dir=erpdiff_dir,
        cfg=cfg,
        pretrained_clb=pretrained["clb"],
        pretrained_rbb=pretrained["rbb"],
        warmup_epochs=warmup_epochs,
        lambda_intra=lambda_intra,
        early_stop_patience=finetune_patience,
        early_stop_min_delta=1e-4,
    )
    finetune_time = time.time() - t1
    total_time = time.time() - t0

    # Collect results from finetune report
    report_path = os.path.join(erpdiff_dir, "finetune_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {}

    report["pretrain_time_s"] = round(pretrain_time, 2)
    report["finetune_time_s"] = round(finetune_time, 2)
    report["total_time_s"] = round(total_time, 2)
    return report


# ============================================================
# Baseline runner
# ============================================================

def _to_numpy_1d(x: Any) -> np.ndarray:
    if x is None:
        return np.empty((0,), dtype=np.float32)
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def _get_loader_meta_arrays(loader: torch.utils.data.DataLoader) -> Dict[str, Optional[np.ndarray]]:
    dataset = getattr(loader, "dataset", None)
    out: Dict[str, Optional[np.ndarray]] = {"runs": None, "reps": None, "flashes": None}
    if dataset is None:
        return out
    for attr, key in (("runs", "runs"), ("reps", "reps"), ("flashes", "flashes")):
        value = getattr(dataset, attr, None)
        if value is not None:
            out[key] = _to_numpy_1d(value)
    return out


def _collect_probs_labels_and_meta_b1(
    model: torch.nn.Module,
    train_result: Dict[str, Any],
    loader: torch.utils.data.DataLoader,
    device: str,
) -> Dict[str, Optional[np.ndarray]]:
    model.load_state_dict(train_result["best_state"])
    probs_pos, labels = collect_probs_and_labels(model, loader, device)
    out = {
        "probs_pos": _to_numpy_1d(probs_pos),
        "labels": _to_numpy_1d(labels),
    }
    out.update(_get_loader_meta_arrays(loader))
    return out


def _collect_probs_labels_and_meta_b2(
    model: torch.nn.Module,
    train_result: Dict[str, Any],
    loader: torch.utils.data.DataLoader,
    device: str,
    b2_config: Dict[str, Any],
) -> Dict[str, Optional[np.ndarray]]:
    n_augments = int(b2_config.get("tta_n_augments", 20))
    ensemble = train_result.get("ensemble")
    if ensemble is not None and len(ensemble) > 0:
        probs_pos, labels = collect_probs_with_ensemble(
            model,
            ensemble,
            loader,
            device,
            use_tta=True,
            n_augments=n_augments,
        )
    else:
        model.load_state_dict(train_result["best_state"])
        probs_pos, labels = collect_probs_with_tta(
            model,
            loader,
            device,
            n_augments=n_augments,
        )
    out = {
        "probs_pos": _to_numpy_1d(probs_pos),
        "labels": _to_numpy_1d(labels),
    }
    out.update(_get_loader_meta_arrays(loader))
    return out


def _build_char_metrics_loader(
    dataset_name: str,
    subject_id: str,
    pkl_path: str,
    cfg: TrainConfig,
    protocol: str,
    test_loader: torch.utils.data.DataLoader,
    b1_config: Dict[str, Any],
    b2_config: Dict[str, Any],
) -> torch.utils.data.DataLoader:
    if dataset_name == "dataset1":
        return test_loader

    _, _, test_ds_meta = build_generic_splits_with_meta(
        subject_id=subject_id,
        pkl_path=pkl_path,
        seed=cfg.seed,
        train_ratio=cfg.train_ratio,
        val_ratio_in_train=cfg.val_ratio_in_train,
        dataset_name=dataset_name,
    )
    batch_size = b1_config["batch_size"] if protocol == "B1" else b2_config["batch_size"]
    return make_loader(test_ds_meta, batch_size, False, cfg)


def _compute_baseline_char_metrics(
    dataset_name: str,
    protocol: str,
    subject_id: str,
    pkl_path: str,
    cfg: TrainConfig,
    model: torch.nn.Module,
    train_result: Dict[str, Any],
    test_loader: torch.utils.data.DataLoader,
    b1_config: Dict[str, Any],
    b2_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute repetition-curve metrics for baseline protocols using a protocol-
    consistent probability source and dataset-specific metadata.
    """
    char_loader = _build_char_metrics_loader(
        dataset_name=dataset_name,
        subject_id=subject_id,
        pkl_path=pkl_path,
        cfg=cfg,
        protocol=protocol,
        test_loader=test_loader,
        b1_config=b1_config,
        b2_config=b2_config,
    )

    if protocol == "B1":
        out = _collect_probs_labels_and_meta_b1(model, train_result, char_loader, cfg.device)
    elif protocol == "B2-Full":
        out = _collect_probs_labels_and_meta_b2(model, train_result, char_loader, cfg.device, b2_config)
    else:
        return {"warning": f"char metrics not configured for protocol={protocol}"}

    probs = _to_numpy_1d(out["probs_pos"])
    labels = _to_numpy_1d(out["labels"])

    if dataset_name == "dataset1":
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        online = obj["online"]
        trial_idx = np.asarray(online["trial_idx"]).reshape(-1)
        stim_code = np.asarray(online["stim_code"]).reshape(-1)
        return compute_char_acc_dataset1_rsvp(
            probs_pos=probs,
            trial_idx=trial_idx,
            stim_code=stim_code,
            labels=labels,
        )

    runs = out["runs"]
    reps = out["reps"]
    flashes = out["flashes"]
    if runs is None or reps is None or flashes is None:
        raise RuntimeError("Missing (run, rep, flash) metadata for char accuracy")

    runs = _to_numpy_1d(runs)
    reps = _to_numpy_1d(reps)
    flashes = _to_numpy_1d(flashes)
    max_k = int(np.max(reps) + 1) if reps.size > 0 else 1

    if dataset_name == "dataset2":
        return compute_char_acc_dataset2(
            probs_pos=probs,
            labels=labels,
            runs=runs,
            reps=reps,
            flashes=flashes,
            max_k=max_k,
        )

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    target_chars = obj.get("target_char", None)
    return compute_char_acc_dataset3(
        probs_pos=probs,
        labels=labels,
        runs=runs,
        reps=reps,
        flashes=flashes,
        max_k=max_k,
        target_chars=target_chars,
    )


def _extract_char_acc_main(dataset_name: str, char_acc: Any) -> Optional[float]:
    if not isinstance(char_acc, dict):
        return None
    if dataset_name == "dataset1":
        v = char_acc.get("char_acc")
    elif dataset_name == "dataset2":
        v = char_acc.get("command_acc_main", char_acc.get("ca_main"))
    elif dataset_name == "dataset3":
        v = char_acc.get(
            "char_acc_main",
            char_acc.get("ca_char_main", char_acc.get("pair_acc_main", char_acc.get("ca_pair_main"))),
        )
    else:
        v = None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _extract_primary_curve(dataset_name: str, char_acc: Any) -> Optional[Dict[str, float]]:
    if not isinstance(char_acc, dict):
        return None
    if dataset_name == "dataset1":
        curve = char_acc.get("char_acc_at_k")
    elif dataset_name == "dataset2":
        curve = char_acc.get("command_acc_at_k", char_acc.get("ca_at_k"))
    elif dataset_name == "dataset3":
        curve = char_acc.get("char_acc_at_k", char_acc.get("ca_char_at_k"))
    else:
        curve = None
    return curve if isinstance(curve, dict) else None


def _extract_pair_curve(char_acc: Any) -> Optional[Dict[str, float]]:
    if not isinstance(char_acc, dict):
        return None
    curve = char_acc.get("pair_acc_at_k", char_acc.get("ca_pair_at_k"))
    return curve if isinstance(curve, dict) else None


def _accumulate_curve_values(accumulator: Dict[int, List[float]], curve: Optional[Dict[str, float]]) -> None:
    if not isinstance(curve, dict):
        return
    for k_str, value in curve.items():
        try:
            accumulator.setdefault(int(k_str), []).append(float(value))
        except Exception:
            continue


def _mean_curve_dict(accumulator: Dict[int, List[float]]) -> Dict[str, float]:
    return {
        str(k): round(float(np.mean(vals)), 6)
        for k, vals in sorted(accumulator.items())
        if vals
    }


def run_baseline_for_dataset(
    model_name: str,
    dataset_name: str,
    predata_dir: str,
    seed: int,
    cfg: TrainConfig,
    out_dir: str,
    protocols: List[str],
    b1_config: Dict[str, Any],
    b2_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train and evaluate a single baseline model on all subjects in a dataset.
    Returns per-subject, per-protocol metrics.
    """
    subject_pkls = discover_subject_pkls(predata_dir)
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "subjects": {},
    }

    for sid, pkl_path in subject_pkls:
        print(f"\n  [{model_name}] Subject: {sid}")
        train_ds, val_ds, test_ds = build_subject_splits(sid, pkl_path, cfg)
        C, T = train_ds.C, train_ds.T

        subject_result = {}

        for protocol in protocols:
            print(f"    Protocol: {protocol}")
            set_global_seed(seed)
            model = get_baseline_model(model_name, C, T, cfg.device)

            if protocol == "B1":
                train_loader = make_loader(train_ds, b1_config["batch_size"], True, cfg)
                val_loader = make_loader(val_ds, b1_config["batch_size"], False, cfg)
                test_loader = make_loader(test_ds, b1_config["batch_size"], False, cfg)

                train_result = train_b1(
                    model, train_loader, val_loader, cfg.device,
                    lr=b1_config["lr"],
                    epochs=b1_config["epochs"],
                    patience=b1_config["patience"],
                    max_class_weight=b1_config["max_class_weight"],
                    model_name=model_name,
                    subject_id=sid,
                )
                eval_result = evaluate_baseline_all_protocols(
                    model, train_result, val_loader, test_loader, cfg.device,
                    protocol="B1",
                    model_name=model_name,
                    subject_id=sid,
                )

            elif protocol == "B2-Full":
                train_loader = make_loader(train_ds, b2_config["batch_size"], True, cfg)
                val_loader = make_loader(val_ds, b2_config["batch_size"], False, cfg)
                test_loader = make_loader(test_ds, b2_config["batch_size"], False, cfg)

                train_result = train_b2(
                    model, train_loader, val_loader, cfg.device,
                    lr=b2_config["lr"],
                    epochs=b2_config["epochs"],
                    patience=b2_config["patience"],
                    use_focal_loss=True,
                    focal_alpha=b2_config["focal_alpha"],
                    focal_gamma=b2_config["focal_gamma"],
                    use_ensemble=True,
                    ensemble_top_k=b2_config["ensemble_top_k"],
                    max_class_weight=b2_config["max_class_weight"],
                    model_name=model_name,
                    subject_id=sid,
                )
                eval_result = evaluate_baseline_all_protocols(
                    model, train_result, val_loader, test_loader, cfg.device,
                    protocol="B2-Full",
                    model_name=model_name,
                    subject_id=sid,
                )
            else:
                print(f"    Unknown protocol: {protocol}, skipping")
                continue

            try:
                eval_result["char_acc"] = _compute_baseline_char_metrics(
                    dataset_name=dataset_name,
                    protocol=protocol,
                    subject_id=sid,
                    pkl_path=pkl_path,
                    cfg=cfg,
                    model=model,
                    train_result=train_result,
                    test_loader=test_loader,
                    b1_config=b1_config,
                    b2_config=b2_config,
                )
            except Exception as e:
                eval_result["char_acc"] = {"error": str(e)}

            eval_result["n_params"] = count_parameters(model)
            subject_result[protocol] = eval_result
            test_m = eval_result.get("test", {})
            print(
                f"    -> Acc={test_m.get('acc', 0):.4f}  "
                f"BalAcc={test_m.get('macro_rec', 0):.4f}  "
                f"F1={test_m.get('macro_f1', 0):.4f}  "
                f"Time={eval_result.get('train_time_s', 0):.1f}s"
            )

        results["subjects"][sid] = subject_result

    # Compute per-protocol average metrics across all subjects
    _METRIC_KEYS = ["acc", "macro_pre", "macro_rec", "macro_f1", "rec_pos", "rec_neg",
                    "pre_pos", "pre_neg", "f1_pos", "f1_neg"]
    for protocol in protocols:
        values = {}
        for sid, sub_res in results["subjects"].items():
            if protocol not in sub_res:
                continue
            test_m = sub_res[protocol].get("test", {})
            for k in _METRIC_KEYS:
                if k in test_m:
                    values.setdefault(k, []).append(test_m[k])
        if not values:
            continue
        n = len(next(iter(values.values())))
        avg = {f"avg_{k}": round(np.mean(v), 6) for k, v in values.items()}
        std = {f"std_{k}": round(np.std(v), 6) for k, v in values.items()}
        avg["n_subjects"] = n
        avg.update(std)
        char_values = []
        curve_accumulator: Dict[int, List[float]] = {}
        pair_curve_accumulator: Dict[int, List[float]] = {}
        for sid, sub_res in results["subjects"].items():
            if protocol not in sub_res:
                continue
            char_acc = sub_res[protocol].get("char_acc", {})
            v = _extract_char_acc_main(dataset_name, char_acc)
            if v is not None:
                char_values.append(v)
            _accumulate_curve_values(curve_accumulator, _extract_primary_curve(dataset_name, char_acc))
            if dataset_name == "dataset3":
                _accumulate_curve_values(pair_curve_accumulator, _extract_pair_curve(char_acc))
        if char_values:
            avg["avg_char_acc_main"] = round(float(np.mean(char_values)), 6)
            avg["std_char_acc_main"] = round(float(np.std(char_values)), 6)
            avg["n_char_subjects"] = len(char_values)
        if curve_accumulator:
            curve_dict = _mean_curve_dict(curve_accumulator)
            if dataset_name == "dataset2":
                avg["avg_command_acc_curve_at_k"] = curve_dict
            else:
                avg["avg_char_acc_curve_at_k"] = curve_dict
        if pair_curve_accumulator:
            avg["avg_pair_acc_curve_at_k"] = _mean_curve_dict(pair_curve_accumulator)
        results[f"summary_{protocol}"] = avg
        char_msg = ""
        if "avg_char_acc_main" in avg:
            char_msg = (
                f" | char_acc={avg['avg_char_acc_main']:.4f}"
                f"+/-{avg['std_char_acc_main']:.4f}"
            )
        print(
            f"\n  [{model_name}][{protocol}][AVERAGE over {n} subjects] "
            f"acc={avg['avg_acc']:.4f}±{std['std_acc']:.4f} | "
            f"bal_acc={avg['avg_macro_rec']:.4f}±{std['std_macro_rec']:.4f} | "
            f"macro_f1={avg['avg_macro_f1']:.4f}±{std['std_macro_f1']:.4f} | "
            f"rec_pos={avg['avg_rec_pos']:.4f}±{std['std_rec_pos']:.4f}"
            f"{char_msg}"
        )

    return results


# ============================================================
# Aggregate and summarize
# ============================================================

def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-subject metrics into mean/std summaries."""
    summary = {}

    for result in all_results:
        model = result.get("model", "erpdiff")
        dataset = result["dataset"]
        seed = result["seed"]

        subjects = result.get("subjects", {})

        # For baselines
        if "subjects" in result:
            protocols = sorted({p for sub_res in subjects.values() for p in sub_res.keys()})
            for protocol in protocols:
                metric_lists = {}
                for sid, sub_res in subjects.items():
                    if protocol not in sub_res:
                        continue
                    test_m = sub_res[protocol].get("test", {})
                    for k in ["acc", "macro_pre", "macro_rec", "macro_f1", "rec_pos"]:
                        metric_lists.setdefault(k, []).append(test_m.get(k, 0))

                if metric_lists:
                    key = f"{model}|{dataset}|{protocol}|seed{seed}"
                    summary[key] = {
                        k: {
                            "mean": round(np.mean(v), 6),
                            "std": round(np.std(v), 6),
                            "n": len(v),
                        }
                        for k, v in metric_lists.items()
                    }

    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ERPDiff vs. Baselines Benchmark")
    parser.add_argument("--datasets", nargs="*", default=["dataset1", "dataset2", "dataset3"],
                        choices=["dataset1", "dataset2", "dataset3"])
    parser.add_argument("--models", nargs="*", default=None,
                        help="Baseline models to run (default: all)")
    parser.add_argument("--exclude-models", nargs="*", default=None,
                        help="Baseline models to exclude (e.g. --exclude-models mocnn ppnn)")
    parser.add_argument("--protocols", nargs="*", default=["B1", "B2-Full"],
                        choices=["B1", "B2-Full"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--skip-erpdiff", action="store_true",
                        help="Skip ERPDiff training (baselines only)")
    parser.add_argument("--run-erpdiff", action="store_true",
                        help="Force run ERPDiff even when --models is specified")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline training (ERPDiff only)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    # B1 config overrides
    parser.add_argument("--b1-epochs", type=int, default=100)
    parser.add_argument("--b1-patience", type=int, default=15)
    parser.add_argument("--b1-lr", type=float, default=0.0001)
    parser.add_argument("--b1-batch-size", type=int, default=128)
    # B2 config overrides
    parser.add_argument("--b2-epochs", type=int, default=500)
    parser.add_argument("--b2-patience", type=int, default=20)
    parser.add_argument("--b2-lr", type=float, default=0.001)
    parser.add_argument("--b2-batch-size", type=int, default=512)
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=0.0,
        help="Upper bound for positive class weight (0 = no cap).",
    )
    # ERPDiff config overrides
    parser.add_argument("--erpdiff-pretrain-epochs", type=int, default=100)
    parser.add_argument("--erpdiff-finetune-epochs", type=int, default=300)
    parser.add_argument("--erpdiff-warmup-epochs", type=int, default=10)
    args = parser.parse_args()

    models = args.models or ALL_BASELINE_MODELS
    if args.exclude_models:
        excluded = set(m.lower() for m in args.exclude_models)
        models = [m for m in models if m.lower() not in excluded]
    # When user explicitly specifies --models, auto-skip ERPDiff unless --run-erpdiff is set
    if args.models and not args.run_erpdiff:
        args.skip_erpdiff = True
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(PROJECT_DIR, f"benchmark_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    b1_config = {
        "lr": args.b1_lr, "epochs": args.b1_epochs,
        "patience": args.b1_patience, "batch_size": args.b1_batch_size,
        "max_class_weight": args.max_class_weight,
    }
    b2_config = {
        "lr": args.b2_lr, "epochs": args.b2_epochs,
        "patience": args.b2_patience, "batch_size": args.b2_batch_size,
        "focal_alpha": 0.70, "focal_gamma": 2.0,
        "ensemble_top_k": 5, "tta_n_augments": 20,
        "max_class_weight": args.max_class_weight,
    }

    all_results = []
    grand_start = time.time()

    for seed in args.seeds:
        for dataset_name in args.datasets:
            predata_dir = DATASET_DIRS.get(dataset_name)
            if predata_dir is None or not os.path.isdir(predata_dir):
                print(f"[WARN] Dataset dir not found: {predata_dir}, skipping {dataset_name}")
                continue

            seed_ds_dir = os.path.join(output_dir, f"seed{seed}", dataset_name)
            os.makedirs(seed_ds_dir, exist_ok=True)

            cfg = TrainConfig()
            cfg.seed = seed
            cfg.dataset = dataset_name
            if args.device:
                cfg.device = args.device
            set_global_seed(seed)

            # ---- ERPDiff ----
            if not args.skip_erpdiff:
                print(f"\n{'#'*60}")
                print(f"# ERPDiff | {dataset_name} | seed={seed}")
                print(f"{'#'*60}")

                erpdiff_cfg = TrainConfig()
                erpdiff_cfg.seed = seed
                erpdiff_cfg.dataset = dataset_name
                erpdiff_cfg.epochs = args.erpdiff_pretrain_epochs
                if args.device:
                    erpdiff_cfg.device = args.device

                erpdiff_result = run_erpdiff_full(
                    dataset_name=dataset_name,
                    predata_dir=predata_dir,
                    seed=seed,
                    out_dir=seed_ds_dir,
                    cfg=erpdiff_cfg,
                    pretrain_patience=14,
                    finetune_patience=35,
                    warmup_epochs=args.erpdiff_warmup_epochs,
                    lambda_intra=0.05,
                )
                erpdiff_result["model"] = "erpdiff"
                erpdiff_result["dataset"] = dataset_name
                erpdiff_result["seed"] = seed

                # Save ERPDiff result
                with open(os.path.join(seed_ds_dir, "erpdiff_result.json"), "w") as f:
                    json.dump(erpdiff_result, f, indent=2, default=str)
                all_results.append(erpdiff_result)

            # ---- Baselines ----
            if not args.skip_baselines:
                for model_name in models:
                    print(f"\n{'#'*60}")
                    print(f"# {model_name} | {dataset_name} | seed={seed}")
                    print(f"{'#'*60}")

                    baseline_result = run_baseline_for_dataset(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        predata_dir=predata_dir,
                        seed=seed,
                        cfg=cfg,
                        out_dir=seed_ds_dir,
                        protocols=args.protocols,
                        b1_config=b1_config,
                        b2_config=b2_config,
                    )

                    # Save per-model result
                    result_path = os.path.join(seed_ds_dir, f"{model_name}_result.json")
                    with open(result_path, "w") as f:
                        json.dump(baseline_result, f, indent=2, default=str)
                    all_results.append(baseline_result)

    total_time = time.time() - grand_start

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print(f"Benchmark complete in {total_time:.1f}s")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    # Build summary table
    summary = _build_summary_table(all_results)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")

    # Print formatted table
    _print_summary_table(summary)


def _build_summary_table(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a summary table from all results."""
    table = {}

    for result in all_results:
        model = result.get("model", "unknown")
        dataset = result.get("dataset", "unknown")
        seed = result.get("seed", 0)

        if model == "erpdiff":
            # ERPDiff results are in the top-level avg_ fields
            key = f"{model}|{dataset}"
            if key not in table:
                table[key] = {"seeds": {}}
            seed_entry = {
                "avg_acc": result.get("avg_acc", 0),
                "avg_macro_f1": result.get("avg_macro_f1", 0),
                "avg_macro_pre": result.get("avg_macro_pre", 0),
                "avg_macro_rec": result.get("avg_macro_rec", 0),
                "total_time_s": result.get("total_time_s", 0),
            }
            for curve_key in [
                "avg_char_acc_curve_at_k",
                "avg_command_acc_curve_at_k",
                "avg_pair_acc_curve_at_k",
            ]:
                curve = result.get(curve_key)
                if isinstance(curve, dict):
                    seed_entry[curve_key] = curve
            table[key]["seeds"][seed] = seed_entry
        else:
            # Baseline results: aggregate across subjects
            subjects = result.get("subjects", {})
            protocols = sorted({p for sub_res in subjects.values() for p in sub_res.keys()})
            for protocol in protocols:
                metric_values = {"acc": [], "macro_f1": [], "macro_pre": [], "macro_rec": [], "rec_pos": []}
                char_values = []
                curve_accumulator = {}
                pair_curve_accumulator = {}
                total_train_time = 0.0

                for sid, sub_res in subjects.items():
                    if protocol not in sub_res:
                        continue
                    test_m = sub_res[protocol].get("test", {})
                    for k in metric_values:
                        metric_values[k].append(test_m.get(k, 0))
                    char_acc = sub_res[protocol].get("char_acc", {})
                    v = _extract_char_acc_main(dataset, char_acc)
                    if v is not None:
                        char_values.append(v)
                    _accumulate_curve_values(curve_accumulator, _extract_primary_curve(dataset, char_acc))
                    if dataset == "dataset3":
                        _accumulate_curve_values(pair_curve_accumulator, _extract_pair_curve(char_acc))
                    total_train_time += sub_res[protocol].get("train_time_s", 0)

                if not metric_values["acc"]:
                    continue

                key = f"{model}|{dataset}|{protocol}"
                if key not in table:
                    table[key] = {"seeds": {}}
                seed_entry = {
                    "avg_acc": round(np.mean(metric_values["acc"]), 6),
                    "avg_macro_f1": round(np.mean(metric_values["macro_f1"]), 6),
                    "avg_macro_pre": round(np.mean(metric_values["macro_pre"]), 6),
                    "avg_macro_rec": round(np.mean(metric_values["macro_rec"]), 6),
                    "avg_rec_pos": round(np.mean(metric_values["rec_pos"]), 6),
                    "total_time_s": round(total_train_time, 2),
                    "n_subjects": len(metric_values["acc"]),
                }
                table[key]["seeds"][seed] = seed_entry
                if char_values:
                    table[key]["seeds"][seed]["avg_char_acc_main"] = round(np.mean(char_values), 6)
                    table[key]["seeds"][seed]["n_char_subjects"] = len(char_values)
                if curve_accumulator:
                    curve_dict = _mean_curve_dict(curve_accumulator)
                    if dataset == "dataset2":
                        table[key]["seeds"][seed]["avg_command_acc_curve_at_k"] = curve_dict
                    else:
                        table[key]["seeds"][seed]["avg_char_acc_curve_at_k"] = curve_dict
                if pair_curve_accumulator:
                    table[key]["seeds"][seed]["avg_pair_acc_curve_at_k"] = _mean_curve_dict(pair_curve_accumulator)

    # Compute cross-seed averages
    for key, entry in table.items():
        seeds_data = entry["seeds"]
        if not seeds_data:
            continue
        metrics_across_seeds = {}
        curve_metrics_across_seeds = {}
        for seed_data in seeds_data.values():
            for k, v in seed_data.items():
                if isinstance(v, (int, float)):
                    metrics_across_seeds.setdefault(k, []).append(v)
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            curve_metrics_across_seeds.setdefault(k, {}).setdefault(sub_k, []).append(sub_v)
        entry["cross_seed_avg"] = {
            k: {"mean": round(np.mean(v), 6), "std": round(np.std(v), 6)}
            for k, v in metrics_across_seeds.items()
        }
        for curve_name, curve_data in curve_metrics_across_seeds.items():
            entry["cross_seed_avg"][curve_name] = {
                sub_k: {"mean": round(np.mean(vals), 6), "std": round(np.std(vals), 6)}
                for sub_k, vals in sorted(curve_data.items(), key=lambda x: int(x[0]))
            }

    return table


def _print_summary_table(summary: Dict[str, Any]):
    """Print a formatted comparison table."""
    print(f"\n{'Model':<25} {'Dataset':<10} {'Protocol':<10} {'Acc':>8} {'BalAcc':>8} {'F1':>8}")
    print("-" * 75)

    for key in sorted(summary.keys()):
        parts = key.split("|")
        model = parts[0]
        dataset = parts[1] if len(parts) > 1 else ""
        protocol = parts[2] if len(parts) > 2 else "full"

        avg = summary[key].get("cross_seed_avg", {})
        acc_m = avg.get("avg_acc", {}).get("mean", 0)
        acc_s = avg.get("avg_acc", {}).get("std", 0)
        rec_m = avg.get("avg_macro_rec", {}).get("mean", 0)
        rec_s = avg.get("avg_macro_rec", {}).get("std", 0)
        f1_m = avg.get("avg_macro_f1", {}).get("mean", 0)
        f1_s = avg.get("avg_macro_f1", {}).get("std", 0)

        print(
            f"{model:<25} {dataset:<10} {protocol:<10} "
            f"{acc_m:.4f}+/-{acc_s:.4f} "
            f"{rec_m:.4f}+/-{rec_s:.4f} "
            f"{f1_m:.4f}+/-{f1_s:.4f}"
        )


if __name__ == "__main__":
    main()
