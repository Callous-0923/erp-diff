# Notes: multi_sigma 实验复现与对比

## Sources

### Source 1: 仓库代码
- 路径: `D:\files\codes\erpdiff`
- 目标:
  - 定位 `multi_sigma` 实现
  - 定位原有方案入口
  - 确认配置与调用链

## Synthesized Findings

### 1. multi_sigma 提案内容
- `multi_sigma.md` 建议把 `TemporalDiffAttn` 的 `sigma_signal/sigma_noise` 从全局共享标量改为按 head 独立的向量。
- 该提案默认只讨论直接替换当前实现，没有设计运行时开关。

### 2. 当前实现落点
- `temporal_diff_attn.py`
  - `TemporalDiffAttn.__init__` 当前在启用 temporal bias 时创建两个标量参数。
  - `_compute_temporal_bias` 当前返回 `[1, 1, T, T]` 形状的 bias。
  - `_sanity_check` 当前用 `.item()` 打印 sigma，仅适用于标量。
- `erpdiff_model.py`
  - `ERPDiff` 构造函数只透传 `use_temporal_bias`，没有 bias 方案选择参数。
  - `load_branch_state` 会把预训练 RBB attention 权重加载到当前模型。
- `erpdiff_rbb_model.py`
  - `RBBPretrainICNN` 只透传 `use_temporal_bias`，没有 bias 方案选择参数。
- `erpdiff_config.py`
  - `TrainConfig` 当前只有 `use_temporal_bias` 开关，没有“shared/per-head”模式字段。
  - YAML 解析只支持 `ablation.use_temporal_bias`。
- `erpdiff_train.py`
  - `_collect_lambda_info` 用 `.item()` 记录 sigma，切到 per-head 后会报错。
  - `parse_args` / `main` 只支持 `--no-temporal-bias`，没有 bias 模式 CLI。
  - `pretrain_stage`、`finetune_stage` 构建模型时没有 bias 模式透传。
- `erpdiff_visualize.py`
  - `plot_lambda_sigma_evolution` 默认把 sigma 当标量时间序列画线。
  - `plot_gaussian_bias_profiles` 用 `max(sig_s, 1.0)`，也假设 sigma 是标量。
  - `plot_attention_heatmaps` 构建 `ERPDiff` 时没有 bias 模式透传。

### 3. 兼容性风险
- 如果把 `sigma_signal/sigma_noise` 直接从标量改成 `[H]`，旧 checkpoint 中同名参数的 shape 会与新模型不一致。
- `load_state_dict(..., strict=False)` 不能自动消化同名参数的 shape mismatch，因此需要显式兼容策略，或者要求 shared / per-head checkpoint 不混用。

### 4. 基线实验信息
- 基线目录：`D:\files\codes\erpdiff\erpdiff_output_260330_d3_9`
- 基线数据集：`dataset3`
- 基线配置：
  - `seed=1`
  - `epochs=300`
  - `pretrain_batch_size=128`
  - `finetune_batch_size=64`
  - `pretrain_lr=0.001`
  - `finetune_lr=0.001`
  - `dropout_p=0.25`
  - `focal_alpha=0.7`
  - `focal_gamma=2.0`
  - `pretrain_weight_decay=0.0005`
  - `finetune_weight_decay=0.001`
  - `use_temporal_bias=true`
  - `use_alpha_gate=true`
  - `lambda_comp=0.1`
  - `comp_margin=0.5`
- finetune 设置：
  - `warmup_epochs=10`
  - `lambda_intra=0.05`
  - `early_stop_patience=35`
  - `early_stop_min_delta=1e-4`
- 可复用 spec：
  - `D:\files\codes\erpdiff\tmp_specs\erpdiff_d3_from_260330_d3_9.yaml`

### 5. 当前实验计划
- 在上述基线参数上增加：
  - `--temporal-bias-sigma-mode per_head`
- 目标输出目录待训练命令确定。

### 6. 实际运行结果
- 新输出目录：
  - `D:\files\codes\erpdiff\erpdiff_output_260411_d3_multi_sigma_from_260330_d3_9`
- 训练日志：
  - `D:\files\codes\erpdiff\erpdiff_output_260411_d3_multi_sigma_from_260330_d3_9\train.log`
- 运行命令要点：
  - 使用 `tmp_specs/erpdiff_d3_from_260330_d3_9.yaml`
  - CLI 覆盖：
    - `--seed 1`
    - `--device cuda`
    - `--warmup-epochs 10`
    - `--lambda-intra 0.05`
    - `--finetune-early-stop-patience 35`
    - `--finetune-early-stop-min-delta 0.0001`
    - `--temporal-bias-sigma-mode per_head`

### 7. 新旧结果对比摘要
- 平均指标
  - `avg_acc`: `0.867647 -> 0.867974` (`+0.000327`)
  - `avg_macro_f1`: `0.691385 -> 0.687908` (`-0.003477`)
  - `avg_macro_pre`: `0.795432 -> 0.807865` (`+0.012433`)
  - `avg_macro_rec`: `0.656667 -> 0.667844` (`+0.011177`)
- `subject_A_Train`
  - `acc`: `0.855229 -> 0.855556`
  - `macro_f1`: `0.650922 -> 0.607849`
  - `macro_rec`: `0.621373 -> 0.586275`
  - `tp/fp/tn/fn`: `138/71/2479/372 -> 93/25/2525/417`
- `subject_B_Train`
  - `acc`: `0.880065 -> 0.880392`
  - `macro_f1`: `0.731847 -> 0.767966`
  - `macro_rec`: `0.691961 -> 0.749412`
  - `tp/fp/tn/fn`: `209/66/2484/301 -> 282/138/2412/228`

### 8. 训练过程差异
- pretrain early stop
  - CLB: `58 -> 58`
  - RBB: `40 -> 36`
- finetune epoch count
  - `subject_A_Train`: `43 -> 40`
  - `subject_B_Train`: `71 -> 164`

### 9. 关于 sigma 日志的解释
- 当前日志记录的是 `sigma_signal/sigma_noise` 参数原值，不是 `clamp(min=1.0)` 后的有效 sigma。
- 因此日志中出现接近 0 或负的极小值时，实际 forward 中仍会被截断到不小于 `1.0`。
