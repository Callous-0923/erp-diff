Now I have the complete current state of the project. Let me produce a thorough summary.

------

## ERPDiff 项目现状总结

### 一、整体架构

ERPDiff 采用两阶段训练的双分支框架，在 ITSEF（Hu et al., 2026, Neural Networks）基础上进行了结构和训练策略的系统性改进。

```
Stage 1 — 跨被试预训练（两个分支独立训练）
  CLB分支：标准ICNN + BSD自蒸馏（CE + KL）
  RBB分支：膨胀ICNN + TemporalDiffAttn + Focal Loss

Stage 2 — 被试特定微调（两分支通过GDCM交互）
  CLB tokens ──────────────────────┐
  RBB tokens → TemporalDiffAttn ──┤── GatedDiffCrossAttn ──→ tails ──→ α加权融合
                                   │
                            互补性正则化约束
```

### 二、模型组件与对应代码

**1. ICNN 骨干网络**（`icnn.py`）— 沿用自 ITSEF

- Temporal Conv (1×64) → Spatial Conv (C×1, depthwise) → Inception#1 → AvgPool → Inception#2
- 输入 `[B, 1, C, T]`，Stem 输出 `[B, 32, 1, T/4]`
- CLB 和 RBB 共享 ICNNTail：AvgPool → FC(288→8) → FC(8→2)

**2. RBB 膨胀 Inception**（`erpdiff_rbb_model.py`）— ERPDiff 新增

- `InceptionBlockDilatedC`：C2 分支用膨胀卷积（k=16, dilation=4），有效感受野 61 时间步，参数量仅为标准 k=64 的 25%
- RBB 与 CLB 在结构上形成非对称

**3. Temporal Differential Self-Attention / TDSA**（`temporal_diff_attn.py`）— ERPDiff 核心

- 差分注意力机制：

  ```
  output = (attn1 − λ·attn2) × V
  ```

  - Q/K/V 投影到 2×d_model，分成两组 (q1,k1) 和 (q2,k2)
  - λ = exp(λ_q1·λ_k1) − exp(λ_q2·λ_k2) + λ_init，可学习标量
  - attn1 捕获信号相关的注意力模式，attn2 捕获噪声模式，做差实现去噪

- 双 sigma 高斯时间偏置（ERP 适配的核心设计）：

  - `sigma_signal`（窄，init=3.0）→ 加到 scores1 → 信号头聚焦 ERP 局部时间窗
  - `sigma_noise`（宽，init=10.0）→ 加到 scores2 → 噪声头捕获全局背景
  - 支持两种模式：`"shared"`（全局 2 个标量）和 `"per_head"`（逐头独立 2H 个参数）
  - per_head 模式下每个 head 自动学习不同的时间尺度 → 多尺度 ERP 建模

- 包含 checkpoint 兼容层（`_load_from_state_dict`）：自动处理 shared↔per_head 之间的权重形状转换

- eval 模式下缓存 attn1/attn2/sigma 值供可解释性分析

**4. Gated Differential Cross-Attention Module / GDCM**（`dcm_diff_cross_attn.py`）— ERPDiff 核心

- 双向差分交叉注意力：CLB←RBB 和 RBB←CLB 各一个 `_DiffCrossAttnDirection`
- 每个方向的内部结构与 TDSA 相同（差分 λ 机制），但 Q 来自一个分支，K/V 来自另一个分支
- α 同步门控：
  - `gate_clb = max(1−α, min_gate)` → 训练初期 CLB 少吸收 RBB 信息（RBB 尚未成熟）
  - `gate_rbb = max(α, min_gate)` → 训练后期 RBB 少吸收 CLB 信息（已充分学习少数类）
  - `min_gate=0.1` 下限防止梯度消失
- 支持 `use_alpha_gate=False` 消融开关（退回为无门控的普通双向交叉注意力）

**5. BSD 自蒸馏**（`erpdiff_train.py` 内）— ERPDiff 新增

- 仅用于 CLB 分支预训练
- 弱增强视图（微缩放+微噪声+微位移）作 teacher，强增强视图（大缩放+大噪声+时间遮蔽+通道丢弃）作 student
- 损失 = CE(student) + β·KL(teacher→student)
- ρ=0.3 降低少数类样本的蒸馏约束，避免 teacher 在少数类上不稳定的预测干扰 student

**6. Focal Loss（修正版）**（`erpdiff_losses.py`）

- 正确实现了 per-class α_t：正类(少数类) α=0.75，负类(多数类) 1−α=0.25
- alpha_weights 用 `register_buffer` 预注册，避免每次 forward 的 CPU→GPU 同步
- 配合 `(1−pt)^γ` 的 hard example mining 同时起效

**7. 互补性保持正则化**（`erpdiff_losses.py`）— ERPDiff 新增

- margin-based soft decorrelation：`L_comp = ReLU(cosine_sim(CLB, RBB) − margin).mean()`
- 只惩罚余弦相似度超过 margin=0.5 的部分，允许基础 EEG 特征共享
- 防止 GDCM 交叉注意力导致两分支特征趋同

**8. 渐进解冻微调**（`erpdiff_train.py`）

- Warmup 阶段：冻结双分支 Stem + TDSA，只训练 Tail + GDCM
- Warmup 结束后：解冻全网络

### 三、相对 ITSEF 的创新点对照

| 维度         | ITSEF                    | ERPDiff                                |
| ------------ | ------------------------ | -------------------------------------- |
| CLB 预训练   | 标准 CE                  | BSD 自蒸馏（CE + KL）                  |
| RBB 结构     | 标准 ICNN（与 CLB 相同） | 膨胀 Inception（非对称）               |
| 分支内注意力 | 无                       | TDSA（差分注意力 + 双sigma高斯偏置）   |
| 分支间融合   | Logits 级 α 加权         | 特征级 GDCM（差分交叉注意力 + α 门控） |
| 特征约束     | 无                       | 互补性正则化                           |
| 微调策略     | 全参数直接微调           | 渐进解冻                               |
| Focal Loss α | 全局常数（无类别区分）   | 正确的 per-class α_t                   |

### 四、论文贡献定位

**贡献 1（核心）**：提出 ERP-Aware Differential Attention 框架——包含分支内 TDSA（带逐头双 sigma 高斯时间偏置的多尺度 ERP 时间建模）和分支间 GDCM（带 α 同步门控的差分交叉注意力融合），统一解决 ERP 信号去噪和双分支互补融合两个问题。

**贡献 2（核心）**：提出训练阶段感知的双分支协同策略——非对称分支结构（标准 Inception CLB + 膨胀 Inception RBB）、BSD 自蒸馏预训练、margin 互补性正则化，从结构、训练、损失三个层面确保双分支互补分化。

**贡献 3（标准）**：在三个 P300 数据集上进行全面实验验证。

### 五、消融实验与当前开关

| CLI 参数             | 配置字段                  | 关闭的模块      |
| -------------------- | ------------------------- | --------------- |
| `--no-temporal-bias` | `use_temporal_bias=False` | TDSA 的高斯偏置 |
| `--no-alpha-gate`    | `use_alpha_gate=False`    | GDCM 的 α 门控  |
| `--lambda-comp 0`    | `lambda_comp=0`           | 互补性正则化    |
| `--clb-bsd-beta 0`   | BSD β=0                   | BSD 自蒸馏      |
| `--warmup-epochs 0`  | warmup=0                  | 渐进解冻        |