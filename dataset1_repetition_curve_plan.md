# Dataset1 1..10 Repetition 曲线实施计划

## 目标
在当前 ERPDiff 项目中，复现并对齐 MOCNN 论文在 Dataset I 上的评估方式，输出 `k=1..10` repetitions 的平均字符/命令识别曲线。

## 论文定义
- MOCNN 将该指标称为 `command recognition accuracy`。
- 对于 repetition `k`，使用前 `k` 轮刺激对应的 EEG classification probabilities 进行 superposition averaging，再得到该字符块的最终识别结果。
- Dataset I 报告 `k=1..10`。

## 当前项目可复用部分
- `erpdiff_train.py` 已能在 test 阶段拿到 `probs_pos` 和 `labels`。
- `_compute_char_metrics_for_subject()` 对 Dataset1 已能读取原始 pkl 中的 `online.trial_idx` 与 `online.stim_code`。
- `compute_char_acc_dataset1_rsvp()` 已实现最终字符级聚合，但目前没有显式输出 1..10 repetition 轨迹。

## 缺口
- 缺少 “每个 `(trial_idx, stim_code)` 是第几次出现” 的 repetition 索引恢复逻辑。
- 缺少 `k=1..10` 的部分和累计聚合评估。
- 缺少 subject 平均后的表格与曲线产物。

## 执行计划

### Phase 1: 明确指标与数据约束
- 确认 Dataset1 online 每个字符块固定 300 个事件。
- 确认每个字符块内 30 个 `stim_code` 各重复 10 次。
- 确认论文里的 command accuracy 等价于当前语境下的字符识别准确率。

### Phase 2: 设计 repetition 索引恢复
- 输入：
  - `trial_idx[N]`
  - `stim_code[N]`
  - `probs_pos[N]`
  - `labels[N]`
- 处理：
  - 对每个字符块 `trial_idx=t` 分组。
  - 在该组内按原始时间顺序遍历。
  - 对每个 `stim_code=s` 计数，得到它的第 `r` 次出现，`r in [1,10]`。
- 输出：
  - `rep_idx[N]`，表示该事件属于第几轮 repetition。

### Phase 3: 设计累计聚合规则
- 对每个字符块 `trial_idx=t`、每个 repetition `k`：
  - 仅使用 `rep_idx <= k` 的事件。
  - 对每个 `stim_code` 聚合其 target 概率，推荐先用求和，再保留均值作为备选。
  - 选择聚合分数最高的 `stim_code` 作为该字符块在 repetition `k` 下的预测字符。
- 真值字符：
  - 在该字符块内，取 `labels==1` 的事件对应的唯一 `stim_code` 作为 ground truth。

### Phase 4: 代码改造点
- 文件：[benchmark_char_metrics.py](/D:/files/codes/erpdiff/benchmark_char_metrics.py)
  - 新增 `compute_char_acc_dataset1_rsvp_curve(...)`
  - 返回：
    - `char_acc`
    - `curve_by_k`，例如 `{1: 0.46, 2: 0.656, ..., 10: 0.933}`
    - `pred_by_trial_by_k`
    - `gt_by_trial`
- 文件：[erpdiff_train.py](/D:/files/codes/erpdiff/erpdiff_train.py)
  - 在 Dataset1 的 test 评估中调用新函数。
  - 将 `curve_by_k` 写入每个 subject 的 `test_metrics["char_acc"]`。
  - 在汇总阶段新增：
    - `avg_char_acc_curve_by_k`
    - `n_char_subjects`
- 可选输出文件：
  - `dataset1_repetition_curve.json`
  - `dataset1_repetition_curve.csv`

### Phase 5: 结果产物设计
- Subject 级 JSON
  - 每个被试 `curve_by_k`
- Overall JSON
  - `avg_curve_by_k`
  - `std_curve_by_k`
- CSV
  - 列：`subject,k,char_acc`
  - 便于后续画图和做论文表格
- 图
  - x 轴：repetition `1..10`
  - y 轴：command/char accuracy
  - 一条主线：ERPDiff 平均曲线
  - 可选对照线：MOCNN / EEG-Inception / EEGNet / HDCA

### Phase 6: 对齐 MOCNN 论文写法
- 表格对齐 Table I：
  - 一行一种方法
  - 一列一个 repetition `k`
  - 单位使用百分比
- 文本对齐：
  - 强调 “with the increase in the number of repetitions, command recognition accuracy increases”
  - 重点比较 `k=1`、`k=2`、`k=3` 的早期解码能力
- 不再把 trial-level `acc` 作为 Dataset1 主指标

### Phase 7: 验证清单
- 检查每个字符块是否恰有 30 个 `stim_code`
- 检查每个 `stim_code` 是否恰出现 10 次
- 检查每个字符块的真值 `stim_code` 是否唯一
- 检查 `k=10` 的最终字符准确率是否与当前 `char_acc` 一致或只存在可解释差异
- 抽查 2 到 3 个被试手工验证 `curve_by_k` 单调上升趋势是否基本成立

## 风险与注意事项
- 若 online 数据某些字符块不是完整 10 轮，需要允许 `k` 的有效样本数变化，并记录 `n_skipped`。
- 若当前 `compute_char_acc_dataset1_rsvp()` 使用的聚合方式不是简单求和，必须先读实现，避免与新曲线口径不一致。
- 若要与 MOCNN 严格对齐，最终阈值不应影响字符级曲线的定义，字符级解码应直接基于概率累计而非二值化标签。

## 推荐执行顺序
1. 先在 `benchmark_char_metrics.py` 中实现 `curve_by_k`
2. 用单个被试 `RSVP_VPfat` 验证输出
3. 接入 `erpdiff_train.py` 的 report
4. 全量跑 Dataset1
5. 生成 Table I 风格 CSV 和图
