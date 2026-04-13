"""
Export repetition-curve CSV/PNG artifacts from finetune reports.

Usage examples:
  python export_repetition_curves.py --report path\\to\\finetune_report.json --dataset dataset1
  python export_repetition_curves.py --report path\\to\\finetune_report.json --dataset dataset2
  python export_repetition_curves.py --report path\\to\\finetune_report.json --dataset dataset3
  python export_repetition_curves.py --report path\\to\\mocnn_result.json --dataset dataset2 --protocol B2-Full
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple


def _resolve_subject_char_acc(
    report: Dict,
    protocol: Optional[str] = None,
) -> Tuple[List[Tuple[str, Dict]], Optional[str]]:
    if isinstance(report.get("subjects"), dict):
        resolved_protocol = protocol
        if resolved_protocol is None:
            candidate_protocols = []
            for sub_res in report["subjects"].values():
                if not isinstance(sub_res, dict):
                    continue
                for proto, payload in sub_res.items():
                    if isinstance(payload, dict) and isinstance(payload.get("char_acc"), dict):
                        candidate_protocols.append(proto)
            if "B1" in candidate_protocols:
                resolved_protocol = "B1"
            elif candidate_protocols:
                resolved_protocol = sorted(set(candidate_protocols))[0]

        rows: List[Tuple[str, Dict]] = []
        for sid, sub_res in sorted(report["subjects"].items()):
            if not isinstance(sub_res, dict) or resolved_protocol not in sub_res:
                continue
            char_acc = sub_res[resolved_protocol].get("char_acc", {})
            if isinstance(char_acc, dict):
                rows.append((sid, char_acc))
        return rows, resolved_protocol

    rows = []
    for sid in sorted(
        k for k in report.keys()
        if isinstance(k, str) and (
            k.startswith("subject")
            or k.startswith("RSVP_")
            or k in {"subject_A_Train", "subject_B_Train", "A", "B"}
        )
    ):
        payload = report.get(sid, {})
        if isinstance(payload, dict):
            char_acc = payload.get("char_acc", {})
            if isinstance(char_acc, dict):
                rows.append((sid, char_acc))
    return rows, None


def _curve_from_char_acc(dataset: str, char_acc: Dict, curve_type: str) -> Dict[str, float]:
    if dataset == "dataset1":
        return char_acc.get("char_acc_at_k", {})
    if dataset == "dataset2":
        return char_acc.get("command_acc_at_k", char_acc.get("ca_at_k", {}))
    if dataset == "dataset3":
        if curve_type == "pair":
            return char_acc.get("pair_acc_at_k", char_acc.get("ca_pair_at_k", {}))
        return char_acc.get("char_acc_at_k", char_acc.get("ca_char_at_k", {}))
    raise ValueError(f"Unsupported dataset for export: {dataset}")


def _protocol_suffix(protocol: Optional[str]) -> str:
    if not protocol:
        return ""
    safe = protocol.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"_{safe}"


def _avg_curve(rows: Iterable[Tuple[str, Dict[str, float]]]) -> Dict[str, float]:
    acc: Dict[int, List[float]] = {}
    for _, curve in rows:
        for k_str, value in curve.items():
            try:
                acc.setdefault(int(k_str), []).append(float(value))
            except Exception:
                continue
    return {str(k): round(sum(vals) / len(vals), 6) for k, vals in sorted(acc.items()) if vals}


def _write_csv(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curve(path: str, title: str, ylabel: str, curve: Dict[str, float]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    xs = [int(k) for k in curve.keys()]
    ys = [float(curve[str(k)]) for k in xs]
    plt.figure(figsize=(7, 4.5), dpi=160)
    plt.plot(xs, ys, marker="o", linewidth=2.2, color="#1f5aa6")
    plt.xticks(xs)
    plt.ylim(max(0.0, min(ys) - 0.1), 1.0)
    plt.xlabel("Repetition (k)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    for x, y in zip(xs, ys):
        plt.text(x, y + 0.012, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return True


def export_dataset1(report_path: str, report: Dict, protocol: Optional[str] = None) -> None:
    out_dir = os.path.dirname(report_path)
    suffix = _protocol_suffix(protocol)
    subject_rows_raw, resolved_protocol = _resolve_subject_char_acc(report, protocol)
    if resolved_protocol and not suffix:
        suffix = _protocol_suffix(resolved_protocol)

    subject_rows: List[Tuple[str, Dict[str, float]]] = []
    csv_rows: List[Dict] = []

    for sid, char_acc in subject_rows_raw:
        curve = _curve_from_char_acc("dataset1", char_acc, "char")
        if not isinstance(curve, dict) or not curve:
            continue
        subject_rows.append((sid, curve))
        for k_str, value in curve.items():
            csv_rows.append({"subject": sid, "k": int(k_str), "char_acc": float(value)})

    avg_curve = _avg_curve(subject_rows)
    _write_csv(os.path.join(out_dir, f"dataset1_repetition_curve{suffix}.csv"), ["subject", "k", "char_acc"], csv_rows)
    _write_csv(
        os.path.join(out_dir, f"dataset1_repetition_curve_avg{suffix}.csv"),
        ["k", "avg_char_acc"],
        [{"k": int(k), "avg_char_acc": float(v)} for k, v in avg_curve.items()],
    )
    _plot_curve(
        os.path.join(out_dir, f"dataset1_repetition_curve_avg{suffix}.png"),
        "Dataset1: Average Character Accuracy vs Repetition",
        "Average Character Accuracy",
        avg_curve,
    )


def export_dataset2(report_path: str, report: Dict, protocol: Optional[str] = None) -> None:
    out_dir = os.path.dirname(report_path)
    suffix = _protocol_suffix(protocol)
    subject_rows_raw, resolved_protocol = _resolve_subject_char_acc(report, protocol)
    if resolved_protocol and not suffix:
        suffix = _protocol_suffix(resolved_protocol)

    subject_rows: List[Tuple[str, Dict[str, float]]] = []
    csv_rows: List[Dict] = []

    for sid, char_acc in subject_rows_raw:
        curve = _curve_from_char_acc("dataset2", char_acc, "char")
        if not isinstance(curve, dict) or not curve:
            continue
        subject_rows.append((sid, curve))
        for k_str, value in curve.items():
            csv_rows.append({"subject": sid, "k": int(k_str), "command_acc": float(value)})

    avg_curve = _avg_curve(subject_rows)
    _write_csv(os.path.join(out_dir, f"dataset2_repetition_curve{suffix}.csv"), ["subject", "k", "command_acc"], csv_rows)
    _write_csv(
        os.path.join(out_dir, f"dataset2_repetition_curve_avg{suffix}.csv"),
        ["k", "avg_command_acc"],
        [{"k": int(k), "avg_command_acc": float(v)} for k, v in avg_curve.items()],
    )
    _plot_curve(
        os.path.join(out_dir, f"dataset2_repetition_curve_avg{suffix}.png"),
        "Dataset2: Average Command Accuracy vs Repetition",
        "Average Command Accuracy",
        avg_curve,
    )


def export_dataset3(report_path: str, report: Dict, protocol: Optional[str] = None) -> None:
    out_dir = os.path.dirname(report_path)
    suffix = _protocol_suffix(protocol)
    subject_rows_raw, resolved_protocol = _resolve_subject_char_acc(report, protocol)
    if resolved_protocol and not suffix:
        suffix = _protocol_suffix(resolved_protocol)

    char_rows: List[Tuple[str, Dict[str, float]]] = []
    pair_rows: List[Tuple[str, Dict[str, float]]] = []
    char_csv_rows: List[Dict] = []
    pair_csv_rows: List[Dict] = []

    for sid, char_acc in subject_rows_raw:
        char_curve = _curve_from_char_acc("dataset3", char_acc, "char")
        pair_curve = _curve_from_char_acc("dataset3", char_acc, "pair")

        if isinstance(char_curve, dict) and char_curve:
            char_rows.append((sid, char_curve))
            for k_str, value in char_curve.items():
                char_csv_rows.append({"subject": sid, "k": int(k_str), "char_acc": float(value)})

        if isinstance(pair_curve, dict) and pair_curve:
            pair_rows.append((sid, pair_curve))
            for k_str, value in pair_curve.items():
                pair_csv_rows.append({"subject": sid, "k": int(k_str), "pair_acc": float(value)})

    avg_char_curve = _avg_curve(char_rows)
    avg_pair_curve = _avg_curve(pair_rows)

    _write_csv(os.path.join(out_dir, f"dataset3_repetition_curve_char{suffix}.csv"), ["subject", "k", "char_acc"], char_csv_rows)
    _write_csv(os.path.join(out_dir, f"dataset3_repetition_curve_pair{suffix}.csv"), ["subject", "k", "pair_acc"], pair_csv_rows)
    _write_csv(
        os.path.join(out_dir, f"dataset3_repetition_curve_avg_char{suffix}.csv"),
        ["k", "avg_char_acc"],
        [{"k": int(k), "avg_char_acc": float(v)} for k, v in avg_char_curve.items()],
    )
    _write_csv(
        os.path.join(out_dir, f"dataset3_repetition_curve_avg_pair{suffix}.csv"),
        ["k", "avg_pair_acc"],
        [{"k": int(k), "avg_pair_acc": float(v)} for k, v in avg_pair_curve.items()],
    )
    _plot_curve(
        os.path.join(out_dir, f"dataset3_repetition_curve_avg_char{suffix}.png"),
        "Dataset3: Average Character Accuracy vs Repetition",
        "Average Character Accuracy",
        avg_char_curve,
    )
    _plot_curve(
        os.path.join(out_dir, f"dataset3_repetition_curve_avg_pair{suffix}.png"),
        "Dataset3: Average Pair Accuracy vs Repetition",
        "Average Pair Accuracy",
        avg_pair_curve,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export repetition-curve CSV/PNG artifacts.")
    parser.add_argument("--report", required=True, help="Path to finetune_report.json")
    parser.add_argument("--dataset", choices=["dataset1", "dataset2", "dataset3"], required=True)
    parser.add_argument(
        "--protocol",
        default=None,
        help="Protocol name for benchmark baseline results (for example: B1 or B2-Full).",
    )
    args = parser.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    if args.dataset == "dataset1":
        export_dataset1(args.report, report, protocol=args.protocol)
    elif args.dataset == "dataset2":
        export_dataset2(args.report, report, protocol=args.protocol)
    else:
        export_dataset3(args.report, report, protocol=args.protocol)


if __name__ == "__main__":
    main()
