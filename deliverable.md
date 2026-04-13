# Deliverable: Benchmark Baseline Repetition Curves

Implemented unified baseline repetition-curve support across `dataset1/2/3`.

## Outcomes
- `benchmark_runner.py` now writes `eval_result["char_acc"]` for both `B1` and `B2-Full`.
- `B2-Full` repetition curves use the same snapshot ensemble + TTA probability source as its test evaluation path.
- Protocol summaries and cross-seed summary tables now emit `avg_*_curve_at_k` whenever a protocol has subject-level `char_acc`.
- `export_repetition_curves.py` now supports benchmark baseline result JSONs and optional `--protocol`, including dataset1 export.

## Verification
- `python -m py_compile` passed for all modified files.
- Python smoke tests confirmed summary aggregation and benchmark export for a synthetic `B2-Full` dataset2 result.
