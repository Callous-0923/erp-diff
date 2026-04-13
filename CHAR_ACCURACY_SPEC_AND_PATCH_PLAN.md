# 字符准确率定义与最小侵入式改造方案（B1）

## 目标
- 给 dataset1 / dataset2 / dataset3 统一定义字符识别准确率（Character Accuracy, CA）。
- 在不改变现有训练和 trial-level 测试逻辑的前提下，新增字符级评估。
- 仅针对 `benchmark_runner.py` 的 B1（单模型）流程。

## 通用定义（B1）
- 对每个 trial 计算目标类概率：`p = softmax(logits)[1]`。
- 对同一字符单元内、同一刺激 code 的分数做累积均值：
  - `S(code, k) = mean(p | 属于该字符单元, code匹配, repetition <= k)`
- `CA@k`：在前 `k` 次重复下，字符预测是否正确的平均值。

---

## Dataset1（RSVP, 30 symbols）

### 依据
- 当前预处理文件：`aa_generate_train_dataset1.py`
- 现有输出字段：`data`, `label`, `trial_idx`（offline/online）

### 字符准确率应定义为
- 以 `trial_idx` 作为字符单元。
- 对该字符单元内 30 个 symbol code 计算 `S(code)`。
- 预测字符：`argmax_code S(code)`。
- 与该字符单元的真实目标字符 code 比较：
  - `CA1 = mean(预测字符 == 真实字符)`

### 当前阻塞
- 当前 pkl 不包含每个 trial 的刺激 code（如 `y_stim`），无法完成上述字符级解码。

### 必要改造（不影响训练/测试）
1. 在 `aa_generate_train_dataset1.py` 的 `preprocess_phase()` 中新增输出字段：
   - `stim_code`: 从 `phase_obj.y_stim` 读取并写入。
2. 保持已有字段不变，新增字段仅用于评估，现有训练代码不会受影响。
3. 重新生成 dataset1 预处理 pkl。

---

## Dataset2（6-command P300）

### 依据
- 当前预处理文件：`aa_generate_train_dataset2.py`
- 数据形状：`data [epoch, 20, 6, C, T]`, `label [epoch, 20, 6]`, `com [epoch]`

### 字符准确率定义
- 每个 epoch 视为一个字符/命令单元。
- 对 6 个 code（1..6）计算 `S(code, k)`。
- 预测：`pred(k) = argmax_{1..6} S(code, k)`。
- 真值：`gt = com - 100`（脚本中写入方式）。
- `CA2@k = mean(pred(k) == gt)`，`k=1..20`。
- 推荐主指标：`CA2@20`，并报告 `CA2@1..20` 曲线。

---

## Dataset3（Wadsworth 6x6, 12 codes）

### 依据
- 当前预处理文件：`aa_generate_train_dataset3.py`
- 数据形状：`data [epoch, 15, 12, C, T]`, `label [epoch, 15, 12]`, `target_char [epoch]`
- code 语义：1..6 为列，7..12 为行；每个字符有 2 个目标 code（1行+1列）。

### 字符准确率定义
- 每个 epoch 视为一个字符单元。
- 计算 12 个 code 的 `S(code, k)`。
- 列预测：`c_hat(k) = argmax_{1..6} S(code, k)`。
- 行预测：`r_hat(k) = argmax_{7..12} S(code, k)`。
- 将 `(r_hat, c_hat)` 映射为字符 `char_hat(k)`，与 `target_char` 比较：
  - `CA3@k = mean(char_hat(k) == target_char)`，`k=1..15`。
- 推荐主指标：`CA3@15`，并报告 `CA3@5`（竞赛常用）。

---

## 代码改造方案（最小侵入，不影响训练/测试）

## 原则
- 训练逻辑保持不变。
- trial-level 指标保持不变。
- 新增“后评估”路径，独立计算字符准确率，写入 JSON。

## 需要新增的模块
- 新建文件：`benchmark_char_metrics.py`
- 提供函数（建议）：
  - `compute_char_acc_dataset2(probs, labels, runs, reps, flashes, max_k)`
  - `compute_char_acc_dataset3(probs, labels, runs, reps, flashes, max_k, target_chars, mapper)`
  - `compute_char_acc_dataset1_rsvp(probs, trial_idx, stim_code, max_k=None)`

## benchmark_runner.py 改造点（B1）
1. 在 B1 测试结束后，增加可选步骤：字符级评估（只读模型输出，不回传梯度）。
2. 字符级结果写入每个模型结果 JSON，例如：
   - `subjects.<sid>.B1.char_acc`
   - 包含：`ca_at_k`、`ca_main`（dataset2=CA@20, dataset3=CA@15, dataset1=CA）
3. Summary 增加字符指标均值/std（如 `avg_char_acc_main`）。

## erpdiff_data.py 改造点（仅评估接口）
- 保持现有 `build_subject_splits()` 不变。
- 新增一个仅供评估的构建接口（建议）：
  - 返回带元数据的数据集（run/rep/flash 或 trial_idx/stim_code）。
- 训练/验证 DataLoader 继续走原函数，互不影响。

## dataset1 额外改造点
- `aa_generate_train_dataset1.py` 输出增加 `stim_code`（从 `y_stim`）。
- 重新预处理 dataset1 后，字符准确率即可按正式定义计算。

---

## 推荐实施顺序（急用版）
1. 先做 dataset2 / dataset3 字符准确率（无需改原始预处理数据）。
2. 再补 dataset1 `stim_code` 并重生成 pkl。
3. 最后在 `benchmark_runner.py` 汇总输出 `char_acc` 到 JSON。

---

## 验证清单
- 训练 loss/acc 曲线与改造前一致。
- B1 原有 trial-level test 指标（acc/macro_rec/macro_f1）与改造前一致。
- 新增 JSON 字段仅增加，不覆盖旧字段。
- dataset2 能输出 `CA@1..20`，dataset3 能输出 `CA@1..15`。
- dataset1 在补齐 `stim_code` 前给出明确提示“缺少字段，无法计算字符准确率”。

