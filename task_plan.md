# Task Plan: multi_sigma 实验复现与对比

## Goal
按照 `erpdiff_output_260330_d3_9` 的参数设置，在 `dataset3` 上运行一轮 `multi_sigma`（`per_head`）实验，并将结果与 `erpdiff_output_260330_d3_9` 对比。

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Read baseline run configuration
- [x] Phase 3: Run multi_sigma experiment
- [x] Phase 4: Compare outputs and deliver

## Implementation
- [x] Step 1: 确认 `erpdiff_output_260330_d3_9` 的配置与评估口径
- [x] Step 2: 确认可复用的 spec 和 CUDA 环境
- [x] Step 3: 使用 `per_head` 开关运行 dataset3
- [x] Step 4: 汇总新旧 `finetune_report.json` 并比较差异

## Key Questions
1. `multi_sigma` 相关实现位于哪些文件，和原有方案的差异点在哪里？
2. 当前方案的配置入口、模型构建入口、训练或推理调用链分别在哪里？
3. 开关最合理的落点是配置文件、命令行参数，还是代码内常量？

## Decisions Made
- 以 `tmp_specs/erpdiff_d3_from_260330_d3_9.yaml` 作为参数基线。
- 只额外打开 `--temporal-bias-sigma-mode per_head`，其余训练参数与基线保持一致。
- 对比以 `finetune_report.json` 中的主体指标为准，辅以 pretrain early stop 和 finetune 轮数信息。

## Errors Encountered
- `rg --files` 在当前环境不可用，改用 PowerShell 文件枚举和文本搜索。
- 当前环境没有 `git` 命令，最终改用 `py_compile`、模块自检和兼容加载脚本完成验证。

## Status
**Completed** - multi_sigma 复现实验已完成，并已与 `erpdiff_output_260330_d3_9` 对比。
