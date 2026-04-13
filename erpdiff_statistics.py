"""
ERPDiff — Statistical Analysis & Significance Testing.

Computes all statistical results needed for the ERPDiff paper:

  1. Per-subject and average metrics (Acc, Macro-F1, Macro-Pre, Macro-Rec)
  2. Paired t-test:        ERPDiff vs each baseline (per-subject accuracy)
  3. Wilcoxon signed-rank:  Non-parametric alternative when n_subjects is small
  4. Friedman test:         Multi-method comparison across all subjects
  5. Nemenyi post-hoc:      Pairwise comparison after Friedman
  6. Ablation significance: Paired t-test between full model and each ablation variant
  7. Effect size:           Cohen's d for each comparison

Usage:
    python erpdiff_statistics.py --run-dir erpdiff_output_260316_d3_1

    # With multi-run results for variance analysis:
    python erpdiff_statistics.py --run-dirs run1/ run2/ run3/ run4/ run5/

    # Compare against baselines (provide JSON with baseline per-subject accuracies):
    python erpdiff_statistics.py --run-dir ... --baselines baselines.json
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# Metric computation
# ============================================================

def compute_metrics_from_confusion(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Compute Acc, Macro-Pre, Macro-Rec, Macro-F1 from confusion counts."""
    eps = 1e-9
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    pre_pos = tp / (tp + fp + eps)
    pre_neg = tn / (tn + fn + eps)
    rec_pos = tp / (tp + fn + eps)
    rec_neg = tn / (tn + fp + eps)

    f1_pos = 2 * pre_pos * rec_pos / (pre_pos + rec_pos + eps)
    f1_neg = 2 * pre_neg * rec_neg / (pre_neg + rec_neg + eps)

    return {
        "acc":       acc,
        "macro_pre": 0.5 * (pre_pos + pre_neg),
        "macro_rec": 0.5 * (rec_pos + rec_neg),
        "macro_f1":  0.5 * (f1_pos + f1_neg),
        "rec_pos":   rec_pos,
        "rec_neg":   rec_neg,
    }


# ============================================================
# Load results
# ============================================================

def load_finetune_report(run_dir: str) -> Dict[str, float]:
    """Load per-subject accuracy from finetune_report.json."""
    path = os.path.join(run_dir, "finetune_report.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    report = json.load(open(path, "r", encoding="utf-8"))
    results = {}
    for key, val in report.items():
        if not isinstance(val, dict):
            continue
        if "test_acc" in val:
            results[key] = val["test_acc"]
        elif "acc" in val:
            results[key] = val["acc"]
    return results


def load_multi_run_results(run_dirs: List[str]) -> Dict[str, List[float]]:
    """Load results from multiple runs, returning {subject: [acc_run1, acc_run2, ...]}."""
    all_results = {}
    for rd in run_dirs:
        report = load_finetune_report(rd)
        for sid, acc in report.items():
            all_results.setdefault(sid, []).append(acc)
    return all_results


# ============================================================
# Statistical tests
# ============================================================

def paired_ttest(accs_a: List[float], accs_b: List[float], method_a: str = "A", method_b: str = "B") -> dict:
    """Paired t-test between two methods' per-subject accuracies."""
    if not HAS_SCIPY:
        return {"error": "scipy not available"}
    a, b = np.array(accs_a), np.array(accs_b)
    if len(a) != len(b) or len(a) < 2:
        return {"error": f"Cannot compare: len(a)={len(a)}, len(b)={len(b)}"}
    t_stat, p_val = sp_stats.ttest_rel(a, b)
    d = cohens_d_paired(a, b)
    return {
        "method_a": method_a,
        "method_b": method_b,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(a - b)),
        "std_diff": float(np.std(a - b, ddof=1)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant_005": bool(p_val < 0.05),
        "significant_001": bool(p_val < 0.01),
        "significant_0001": bool(p_val < 0.001),
        "cohens_d": float(d),
        "n_subjects": len(a),
    }


def wilcoxon_test(accs_a: List[float], accs_b: List[float], method_a: str = "A", method_b: str = "B") -> dict:
    """Wilcoxon signed-rank test (non-parametric alternative to paired t-test)."""
    if not HAS_SCIPY:
        return {"error": "scipy not available"}
    a, b = np.array(accs_a), np.array(accs_b)
    diff = a - b
    if np.all(diff == 0):
        return {"method_a": method_a, "method_b": method_b, "p_value": 1.0, "note": "All differences are zero."}
    try:
        stat, p_val = sp_stats.wilcoxon(a, b)
    except ValueError as e:
        return {"error": str(e)}
    return {
        "method_a": method_a,
        "method_b": method_b,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "w_statistic": float(stat),
        "p_value": float(p_val),
        "significant_005": bool(p_val < 0.05),
        "n_subjects": len(a),
    }


def friedman_test(all_method_accs: Dict[str, List[float]]) -> dict:
    """
    Friedman test across multiple methods.

    Args:
        all_method_accs: {method_name: [acc_sub1, acc_sub2, ...], ...}
            All lists must have the same length (same subjects in same order).
    """
    if not HAS_SCIPY:
        return {"error": "scipy not available"}
    methods = list(all_method_accs.keys())
    arrays = [np.array(all_method_accs[m]) for m in methods]

    # Verify all same length
    lengths = set(len(a) for a in arrays)
    if len(lengths) != 1:
        return {"error": f"Inconsistent subject counts: {lengths}"}

    stat, p_val = sp_stats.friedmanchisquare(*arrays)

    # Compute average ranks
    n_subjects = len(arrays[0])
    n_methods = len(methods)
    ranks = np.zeros((n_subjects, n_methods))
    for i in range(n_subjects):
        row = np.array([arrays[j][i] for j in range(n_methods)])
        ranks[i] = sp_stats.rankdata(-row)  # Higher acc = lower rank (rank 1 = best)
    avg_ranks = ranks.mean(axis=0)

    rank_dict = {methods[j]: float(avg_ranks[j]) for j in range(n_methods)}

    return {
        "chi2_statistic": float(stat),
        "p_value": float(p_val),
        "significant_005": bool(p_val < 0.05),
        "n_subjects": n_subjects,
        "n_methods": n_methods,
        "average_ranks": rank_dict,
    }


def nemenyi_critical_difference(n_methods: int, n_subjects: int, alpha: float = 0.05) -> float:
    """
    Compute the Nemenyi post-hoc test critical difference.
    CD = q_alpha * sqrt(n_methods * (n_methods + 1) / (6 * n_subjects))
    """
    # q_alpha values for Nemenyi test (Demšar 2006, Table 5)
    # Index: n_methods - 2 (for methods 2..10)
    q_table_005 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q = q_table_005.get(n_methods)
    if q is None:
        return float("nan")
    cd = q * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_subjects))
    return cd


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    diff = a - b
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-9))


# ============================================================
# Multi-run variance analysis
# ============================================================

def multi_run_variance_analysis(run_dirs: List[str]) -> dict:
    """
    Analyze variance across multiple independent runs.
    Reports mean ± std of per-subject accuracy across runs.
    """
    all_results = load_multi_run_results(run_dirs)
    summary = {}
    for sid, accs in all_results.items():
        summary[sid] = {
            "n_runs": len(accs),
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "min": float(np.min(accs)),
            "max": float(np.max(accs)),
            "values": [float(v) for v in accs],
        }
    # Overall average
    all_accs = [np.mean(accs) for accs in all_results.values()]
    summary["__overall__"] = {
        "n_subjects": len(all_results),
        "n_runs": len(run_dirs),
        "mean": float(np.mean(all_accs)),
        "std": float(np.std(all_accs, ddof=1)) if len(all_accs) > 1 else 0.0,
    }
    return summary


# ============================================================
# Full statistical report
# ============================================================

def generate_full_report(
    erpdiff_dir: str,
    baselines: Optional[Dict[str, Dict[str, float]]] = None,
    ablation_dirs: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> dict:
    """
    Generate a complete statistical report.

    Args:
        erpdiff_dir:    Path to ERPDiff training output.
        baselines:      {method_name: {subject_id: accuracy, ...}, ...}
        ablation_dirs:  {variant_name: run_dir_path, ...}
        save_path:      Where to save the JSON report.

    Returns:
        Full report dictionary.
    """
    report = {"erpdiff": {}}

    # 1. Load ERPDiff results
    erpdiff_accs = load_finetune_report(erpdiff_dir)
    subjects = sorted(erpdiff_accs.keys())
    erpdiff_vals = [erpdiff_accs[s] for s in subjects]

    report["erpdiff"] = {
        "per_subject": {s: float(erpdiff_accs[s]) for s in subjects},
        "mean_acc": float(np.mean(erpdiff_vals)),
        "std_acc": float(np.std(erpdiff_vals, ddof=1)) if len(erpdiff_vals) > 1 else 0.0,
    }

    # 2. Pairwise comparisons with baselines
    if baselines:
        report["pairwise_ttest"] = {}
        report["pairwise_wilcoxon"] = {}
        all_method_accs = {"ERPDiff": erpdiff_vals}

        for method, method_accs in baselines.items():
            baseline_vals = [method_accs.get(s, 0) for s in subjects]
            all_method_accs[method] = baseline_vals

            report["pairwise_ttest"][f"ERPDiff_vs_{method}"] = paired_ttest(
                erpdiff_vals, baseline_vals, "ERPDiff", method
            )
            report["pairwise_wilcoxon"][f"ERPDiff_vs_{method}"] = wilcoxon_test(
                erpdiff_vals, baseline_vals, "ERPDiff", method
            )

        # 3. Friedman test
        report["friedman"] = friedman_test(all_method_accs)

        # 4. Nemenyi CD
        if report["friedman"].get("significant_005"):
            n_m = report["friedman"]["n_methods"]
            n_s = report["friedman"]["n_subjects"]
            cd = nemenyi_critical_difference(n_m, n_s)
            report["nemenyi"] = {
                "critical_difference": float(cd),
                "n_methods": n_m,
                "n_subjects": n_s,
                "alpha": 0.05,
                "average_ranks": report["friedman"]["average_ranks"],
            }

    # 5. Ablation significance
    if ablation_dirs:
        report["ablation"] = {}
        for variant, abl_dir in ablation_dirs.items():
            try:
                abl_accs = load_finetune_report(abl_dir)
                abl_vals = [abl_accs.get(s, 0) for s in subjects]
                report["ablation"][variant] = {
                    "mean_acc": float(np.mean(abl_vals)),
                    "std_acc": float(np.std(abl_vals, ddof=1)) if len(abl_vals) > 1 else 0.0,
                    "ttest_vs_full": paired_ttest(erpdiff_vals, abl_vals, "ERPDiff (full)", variant),
                }
            except Exception as e:
                report["ablation"][variant] = {"error": str(e)}

    # 6. Save
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Saved] Statistical report: {save_path}")

    return report


def print_report_summary(report: dict):
    """Pretty-print key results from the statistical report."""
    print("\n" + "=" * 70)
    print("  ERPDiff — Statistical Analysis Summary")
    print("=" * 70)

    erp = report.get("erpdiff", {})
    print(f"\n  ERPDiff Average Accuracy: {erp.get('mean_acc', 0) * 100:.2f}% ± {erp.get('std_acc', 0) * 100:.2f}%")

    if "pairwise_ttest" in report:
        print("\n  Paired t-test (ERPDiff vs Baselines):")
        print(f"  {'Comparison':<35} {'Δ Acc':>8} {'p-value':>10} {'Sig.':>6} {'Cohen d':>8}")
        print("  " + "-" * 67)
        for name, res in report["pairwise_ttest"].items():
            if "error" in res:
                print(f"  {name:<35} ERROR: {res['error']}")
                continue
            diff = res["mean_diff"] * 100
            p = res["p_value"]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            d = res["cohens_d"]
            print(f"  {name:<35} {diff:>+7.2f}% {p:>10.4f} {sig:>6} {d:>8.3f}")

    if "friedman" in report:
        fr = report["friedman"]
        print(f"\n  Friedman Test:  χ²={fr.get('chi2_statistic', 0):.3f},  "
              f"p={fr.get('p_value', 1):.4f},  "
              f"{'SIGNIFICANT' if fr.get('significant_005') else 'not significant'}")
        if "average_ranks" in fr:
            print("  Average Ranks:")
            for method, rank in sorted(fr["average_ranks"].items(), key=lambda x: x[1]):
                print(f"    {method:<25} {rank:.2f}")

    if "nemenyi" in report:
        cd = report["nemenyi"]["critical_difference"]
        print(f"\n  Nemenyi Post-hoc CD = {cd:.3f} (α=0.05)")

    if "ablation" in report:
        print("\n  Ablation Study (paired t-test vs full model):")
        print(f"  {'Variant':<30} {'Acc':>8} {'Δ':>8} {'p-value':>10} {'Sig.':>6}")
        print("  " + "-" * 62)
        full_acc = erp.get("mean_acc", 0) * 100
        for variant, res in report["ablation"].items():
            if "error" in res:
                print(f"  {variant:<30} ERROR: {res['error']}")
                continue
            acc = res["mean_acc"] * 100
            t = res.get("ttest_vs_full", {})
            p = t.get("p_value", 1)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            print(f"  {variant:<30} {acc:>7.2f}% {acc - full_acc:>+7.2f}% {p:>10.4f} {sig:>6}")

    print("\n" + "=" * 70)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ERPDiff statistical analysis.")
    parser.add_argument("--run-dir", default=None, help="Single run output directory.")
    parser.add_argument("--run-dirs", nargs="*", default=None, help="Multiple run dirs for variance analysis.")
    parser.add_argument("--baselines", default=None,
                        help="JSON file with baseline results: {method: {subject: acc, ...}, ...}")
    parser.add_argument("--ablation-dirs", default=None,
                        help="JSON file mapping ablation variant names to run dirs.")
    parser.add_argument("--save", default=None, help="Path to save statistical report JSON.")
    args = parser.parse_args()

    if not HAS_SCIPY:
        print("[Warning] scipy not installed — statistical tests will be unavailable.")
        print("  Install with: pip install scipy")

    # Multi-run variance analysis
    if args.run_dirs:
        print("\n[Multi-run Variance Analysis]")
        var_report = multi_run_variance_analysis(args.run_dirs)
        overall = var_report.get("__overall__", {})
        print(f"  {overall.get('n_runs', 0)} runs × {overall.get('n_subjects', 0)} subjects")
        print(f"  Overall: {overall.get('mean', 0) * 100:.2f}% ± {overall.get('std', 0) * 100:.2f}%")
        for sid, info in sorted(var_report.items()):
            if sid.startswith("__"):
                continue
            print(f"  {sid}: {info['mean'] * 100:.2f}% ± {info['std'] * 100:.2f}%")

    # Single-run detailed analysis
    if args.run_dir:
        baselines = None
        if args.baselines and os.path.exists(args.baselines):
            with open(args.baselines, "r") as f:
                baselines = json.load(f)

        ablation_dirs = None
        if args.ablation_dirs and os.path.exists(args.ablation_dirs):
            with open(args.ablation_dirs, "r") as f:
                ablation_dirs = json.load(f)

        save_path = args.save or os.path.join(args.run_dir, "statistical_report.json")
        report = generate_full_report(args.run_dir, baselines, ablation_dirs, save_path)
        print_report_summary(report)


if __name__ == "__main__":
    main()
