# Dataset2 and Dataset3 1..10 Repetition 曲线实施计划

## 目标
在当前 ERPDiff 项目中，为 dataset2 和 dataset3 建立与 MOCNN 论文一致的 `1..10` repetition 字符/命令识别曲线报告流程，并输出可直接用于论文表格和图的 JSON / CSV / PNG 产物。

## 论文口径
- MOCNN 在 Datasets I–III 上都报告 `command recognition accuracy` 随 repetition 增加的结果。
- `k` 的定义是：使用前 `k` 轮 repetition 的分类概率，通过 superposition averaging 得到最终识别结果。
- 因此目标不是单个 `k=max` 的字符准确率，而是完整的 `k=1..10` 累计曲线。

## 数据集对应关系

### Dataset2
- 原始任务：6 选 1 命令识别。
- 原始文件字段：
  - `data`: `34 x samples` 连续 EEG
  - `events`: 刺激时间点
  - `stimuli`: `1..6`
  - `target`: 当前 run 的目标图像编号
  - `targets_count`: 用户数到的目标闪烁次数
- 当前预处理布局：
  - `data`: `[run, repeat, flash, C, T]`
  - `label`: `[run, repeat, flash]`
- 当前 repetition 总数：20
- 与论文对齐的报告区间：前 10 次 repetition

### Dataset3
- 原始任务：6x6 P300 speller 字符识别。
- 原始文件字段：
  - `Signal`, `Flashing`, `StimulusCode`, `StimulusType`, `TargetChar`
- 当前预处理布局：
  - `data`: `[epoch, repeat, code, C, T]`
  - `label`: `[epoch, repeat, code]`
  - `target_char`: `[epoch]`
- 当前 repetition 总数：15
- 与论文对齐的报告区间：前 10 次 repetition

## 当前项目现状

### 已有能力
- `compute_char_acc_dataset2()` 已能输出 `ca_at_k`
- `compute_char_acc_dataset3()` 已能输出：
  - `ca_pair_at_k`
  - `ca_pair_main`
  - `ca_char_at_k`
  - `ca_char_main`
- `build_generic_splits_with_meta()` 已能在 test 集上恢复 `run / rep / flash` 元数据

### 仍缺部分
- dataset2/3 的训练报告未统一汇总 repetition 曲线
- benchmark 汇总未统一导出 Dataset2/3 的 `avg_curve_at_k`
- 没有生成 Table II / Table III 风格 CSV
- 没有生成可直接插图的平均曲线 PNG
- metric naming 还不够统一，不利于论文写作

## 详细执行计划

### Phase 1: 明确每个数据集的主曲线定义

#### Dataset2 主指标
- 主指标名称：`command_acc_at_k`
- 含义：对每个 run，在前 `k` 次 repetition 内，对 6 个 flash 的 target 概率做累计聚合，选出分数最高的那个命令。
- 现有实现映射：
  - `compute_char_acc_dataset2()["ca_at_k"]`
- 计划动作：
  - 保留现有算法
  - 在报告层将 `ca_at_k` 规范别名为 `command_acc_at_k`

#### Dataset3 主指标
- 主指标名称建议拆成两条：
  - `command_acc_at_k`：row/column 联合配对正确率
  - `char_acc_at_k`：最终字符正确率
- 当前实现映射：
  - `ca_pair_at_k` 对应命令/配对正确率
  - `ca_char_at_k` 对应字符正确率
- 计划动作：
  - 在论文主表中优先使用 `char_acc_at_k`
  - 同时保留 `pair_acc_at_k` 作为补充分析

### Phase 2: 统一 repetition 上限到 1..10
- dataset2 当前最大 repetition 为 20
- dataset3 当前最大 repetition 为 15
- MOCNN 对论文表格使用统一 repetition 轴
- 计划动作：
  - 在 `benchmark_char_metrics.py` 为 dataset2/3 增加可配置参数 `report_max_k=10`
  - 曲线计算仍可保留完整 repetition，但主报告和导出默认截取 `1..10`

### Phase 3: 统一 metric 命名和返回结构

#### Dataset2 返回结构目标
```json
{
  "metric": "command_acc_dataset2",
  "command_acc_at_k": {"1": ..., "2": ..., "...": ..., "10": ...},
  "command_acc_main": ...,
  "main_k": 10,
  "n_epochs": ...
}
```

#### Dataset3 返回结构目标
```json
{
  "metric": "char_acc_dataset3",
  "pair_acc_at_k": {"1": ..., "...": "10"},
  "pair_acc_main": ...,
  "char_acc_at_k": {"1": ..., "...": "10"},
  "char_acc_main": ...,
  "main_k": 10,
  "n_epochs": ...
}
```

#### 具体动作
- 在 [benchmark_char_metrics.py](/D:/files/codes/erpdiff/benchmark_char_metrics.py) 中：
  - 为 dataset2/3 增加统一别名字段
  - 保留旧字段兼容历史代码

### Phase 4: 扩展 ERPDiff finetune 报告
- 文件：[erpdiff_train.py](/D:/files/codes/erpdiff/erpdiff_train.py)
- 计划动作：
  - dataset2：
    - 汇总 `avg_command_acc_curve_at_k`
  - dataset3：
    - 汇总 `avg_pair_acc_curve_at_k`
    - 汇总 `avg_char_acc_curve_at_k`
  - 将 `main_k=10` 写入报告

### Phase 5: 扩展 benchmark 汇总
- 文件：[benchmark_runner.py](/D:/files/codes/erpdiff/benchmark_runner.py)
- 计划动作：
  - 在 `summary_B1` 中按 dataset 名称分别汇总：
    - dataset2: `avg_command_acc_curve_at_k`
    - dataset3: `avg_pair_acc_curve_at_k`, `avg_char_acc_curve_at_k`
  - 保持 dataset1 / dataset2 / dataset3 三套命名一致

### Phase 6: 导出论文表格所需文件

#### Dataset2
- `dataset2_repetition_curve.csv`
  - 列：`subject,k,command_acc`
- `dataset2_repetition_curve_avg.csv`
  - 列：`k,avg_command_acc`

#### Dataset3
- `dataset3_repetition_curve_char.csv`
  - 列：`subject,k,char_acc`
- `dataset3_repetition_curve_pair.csv`
  - 列：`subject,k,pair_acc`
- `dataset3_repetition_curve_avg_char.csv`
  - 列：`k,avg_char_acc`
- `dataset3_repetition_curve_avg_pair.csv`
  - 列：`k,avg_pair_acc`

### Phase 7: 生成图像产物

#### Dataset2
- 一张平均命令识别曲线图
  - x 轴：`1..10`
  - y 轴：`Average Command Accuracy`

#### Dataset3
- 两张图
  - `Average Character Accuracy vs Repetition`
  - `Average Pair Accuracy vs Repetition`

### Phase 8: 验证清单

#### Dataset2
- 检查每个 run 是否包含 6 个 flash × 20 repetitions
- 检查每个 repetition 是否只有 1 个 target
- 检查 `k=10` 是否与旧 `ca_at_k["10"]` 一致
- 抽查几位被试曲线是否大体单调上升

#### Dataset3
- 检查每个 epoch 是否包含 12 个 code × 15 repetitions
- 检查每个 epoch 是否恰有 1 行 + 1 列 target
- 检查 `k=10` 的 `char_acc_at_k["10"]` 与原字符指标一致
- 检查 `pair_acc` 与 `char_acc` 的差异是否合理

### Phase 9: 论文写法对齐
- dataset2 对齐 Table II 风格：
  - 只展示 `1..10` 命令识别率
- dataset3 对齐 Table III 风格：
  - 主表展示 `1..10` 字符识别率
  - 附录或补充材料展示 `pair_acc_at_k`
- 正文解释：
  - `k` 表示前 `k` 轮 repetition 的分类概率累积
  - 强调早期 repetition (`k=1,2,3`) 的差异

## 推荐实施顺序
1. 核对并统一 dataset2/3 指标函数字段名
2. 在 `erpdiff_train.py` 聚合 avg curve
3. 在 `benchmark_runner.py` 聚合 avg curve
4. 先用一个 dataset2 历史输出做 smoke test
5. 再用一个 dataset3 历史输出做 smoke test
6. 最后全量 rerun 或从已有结果重导出表格和图

## 风险与注意事项
- dataset2 和 dataset3 当前算法层面已经支持曲线，不一定需要重训；很多情况下只需重导出或重评估。
- dataset3 同时有 pair-level 和 char-level 两种口径，论文主指标必须提前定死，避免后续比较混乱。
- 如果历史 `json` 没有保存 test set 概率，则不能纯重导出，只能重新跑评估阶段。
