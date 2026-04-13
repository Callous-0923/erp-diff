# multi_sigma 开关改造计划

## 目标
在不移除当前 shared-sigma 方案的前提下，增加一个可配置开关，在以下两种实现之间切换：

- `shared`：保留当前方案，`sigma_signal/sigma_noise` 为全局共享标量
- `per_head`：启用 `multi_sigma.md` 中的 multi-sigma 方案，`sigma_signal/sigma_noise` 为按 head 独立的向量

## 建议的实现顺序

### 1. 先定义统一的配置接口
- 在 `TrainConfig` 中新增一个模式字段，建议命名为 `temporal_bias_sigma_mode`，取值 `shared` / `per_head`。
- 保持 `use_temporal_bias` 继续负责“开 / 关 temporal bias”，新字段只负责“bias 开启时用哪种 sigma 方案”。
- YAML 中增加对应配置项，建议放在 `ablation` 段内，避免再造新的配置层级。
- CLI 中增加对应参数，建议显式覆盖 YAML。

### 2. 把模式沿构造链透传到模型
- `ERPDiff` 增加该参数并传给 `TemporalDiffAttn`。
- `RBBPretrainICNN` 增加该参数并传给 `TemporalDiffAttn`。
- `pretrain_stage`、`finetune_stage`、可视化脚本里的模型构建都要一起透传，保证训练、评估、画图使用同一模式。

### 3. 在 `TemporalDiffAttn` 内同时支持两套参数形态
- 在 `TemporalDiffAttn.__init__` 中根据模式创建标量参数或长度为 `num_heads` 的向量参数。
- 在 `_compute_temporal_bias` 中根据模式返回：
  - `shared`：`[1, 1, T, T]`
  - `per_head`：`[1, H, T, T]`
- `forward` 保持现有加法逻辑即可，广播会自动处理。
- `_sanity_check` 的打印逻辑也要分模式处理，避免 `.item()` 在向量模式下报错。

### 4. 修正日志与可视化对 sigma 标量的假设
- `_collect_lambda_info` 需要按模式记录：
  - `shared`：继续记录 float
  - `per_head`：记录 list，或同时记录原始 list 和均值摘要
- `plot_lambda_sigma_evolution` 需要决定 per-head 数据如何展示，建议：
  - 每个 head 一条线；或
  - 默认画均值，再可选画各 head 虚线
- `plot_gaussian_bias_profiles` 需要遍历 per-head sigma，而不是直接把 sigma 当标量。
- `plot_attention_heatmaps` 构建模型时要传入相同模式，否则加载 checkpoint 会和训练配置不一致。

### 5. 处理 checkpoint 兼容问题
- 这是这次改造里最容易踩坑的部分。
- 建议二选一：
  - 严格模式：shared / per-head checkpoint 不混用，加载前检查模式并给出明确报错；
  - 兼容模式：加载 old shared checkpoint 到 per-head 模型时，把标量复制为每个 head 的初值；反向加载时取均值或第一个 head。
- 如果不做兼容层，至少要把模式写入日志 / checkpoint 对应配置，并在加载入口做一致性校验。

### 6. 补最小验证闭环
- 覆盖 `use_temporal_bias=False`
- 覆盖 `use_temporal_bias=True + shared`
- 覆盖 `use_temporal_bias=True + per_head`
- 重点验证：
  - forward shape 正常
  - 日志导出不报错
  - 可视化脚本能读取新日志
  - checkpoint 加载行为符合预期

## 建议优先级

1. 先改配置接口和模型透传
2. 再改 `TemporalDiffAttn` 双模式实现
3. 然后处理日志与可视化
4. 最后补 checkpoint 兼容与验证

## 预计改动文件

- `D:\files\codes\erpdiff\erpdiff_config.py`
- `D:\files\codes\erpdiff\erpdiff_train.py`
- `D:\files\codes\erpdiff\erpdiff_model.py`
- `D:\files\codes\erpdiff\erpdiff_rbb_model.py`
- `D:\files\codes\erpdiff\temporal_diff_attn.py`
- `D:\files\codes\erpdiff\erpdiff_visualize.py`
