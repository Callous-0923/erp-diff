------

## 改动总览

**改什么**：将 `temporal_diff_attn.py` 中的双 sigma 高斯时间偏置从"全局共享 2 个标量"升级为"逐头独立 2H 个参数"。

**为什么**：

- 当前全局共享时，4 个 head 被迫用同一个时间尺度关注 ERP，容易导致 sigma 坍缩（你之前训练中已观察到这个问题）
- 逐头独立后，每个 head 自由学习不同的时间尺度——有的聚焦 P300 峰值（σ≈2-3），有的捕获晚期慢电位（σ≈8-10）
- 论文叙事从"加了两个参数"升级为"多尺度 ERP 时间建模"
- 训练完成后可以展示 H 个 sigma 值的分化情况，作为可解释性证据

**改几行**：约 15 行改动。

**影响哪些文件**：只改 `temporal_diff_attn.py` 一个文件。`erpdiff_model.py`、`erpdiff_rbb_model.py`、`dcm_diff_cross_attn.py`、`erpdiff_train.py` 均不需要改。

------

## 具体改动（3 处）

### 改动 ①：`__init__` — 参数声明

**当前**（全局共享，或者你合入了之前版本后的状态）：

```python
self.sigma_signal = nn.Parameter(torch.tensor(3.0))       # 1个标量
self.sigma_noise  = nn.Parameter(torch.tensor(10.0))      # 1个标量
```

**改为**（逐头独立）：

```python
self.sigma_signal = nn.Parameter(torch.full((num_heads,), 3.0))    # [H]
self.sigma_noise  = nn.Parameter(torch.full((num_heads,), 10.0))   # [H]
```

### 改动 ②：`_compute_temporal_bias` — 偏置计算

**当前**：

```python
def _compute_temporal_bias(self, t_steps, dtype, device):
    pos = torch.arange(t_steps, dtype=dtype, device=device)
    rel_dist = pos.unsqueeze(0) - pos.unsqueeze(1)              # [T, T]
    sigma_s = self.sigma_signal.clamp(min=1.0)                  # 标量
    sigma_n = self.sigma_noise.clamp(min=1.0)                   # 标量
    bias_signal = -0.5 * (rel_dist / sigma_s) ** 2              # [T, T]
    bias_noise  = -0.5 * (rel_dist / sigma_n) ** 2              # [T, T]
    return bias_signal.unsqueeze(0).unsqueeze(0),               # [1, 1, T, T]
           bias_noise.unsqueeze(0).unsqueeze(0)
```

**改为**：

```python
def _compute_temporal_bias(self, t_steps, dtype, device):
    pos = torch.arange(t_steps, dtype=dtype, device=device)
    rel_dist = pos.unsqueeze(0) - pos.unsqueeze(1)              # [T, T]
    sigma_s = self.sigma_signal.clamp(min=1.0)                  # [H]
    sigma_n = self.sigma_noise.clamp(min=1.0)                   # [H]
    # [H, 1, 1] × [T, T] → [H, T, T]
    bias_signal = -0.5 * (rel_dist.unsqueeze(0) / sigma_s.view(-1, 1, 1)) ** 2
    bias_noise  = -0.5 * (rel_dist.unsqueeze(0) / sigma_n.view(-1, 1, 1)) ** 2
    return bias_signal.unsqueeze(0),                            # [1, H, T, T]
           bias_noise.unsqueeze(0)                              # [1, H, T, T]
```

### 改动 ③：`forward` 中 bias 的加法广播

**当前**（如果用 `[1, 1, T, T]`）：

```python
scores1 = scores1 + bias_signal     # [B,H,T,T] + [1,1,T,T] → OK
scores2 = scores2 + bias_noise
```

**改为**（用 `[1, H, T, T]`）：

```python
scores1 = scores1 + bias_signal     # [B,H,T,T] + [1,H,T,T] → OK，每个head不同的bias
scores2 = scores2 + bias_noise
```

这一步实际上**代码不需要改**——PyTorch 的广播规则下 `[B,H,T,T] + [1,H,T,T]` 自动按 head 维度对齐。但语义变了：之前每个 head 加的是同一个偏置，现在每个 head 加的是自己独有的偏置。

------

## 对 `_collect_lambda_info` 记录的影响

`erpdiff_train.py` 中的 `_collect_lambda_info` 当前记录：

```python
info["sigma_signal"] = float(attn.sigma_signal.item())
info["sigma_noise"]  = float(attn.sigma_noise.item())
```

改为逐头后 `.item()` 会报错（因为不再是标量），需要改为：

```python
info["sigma_signal"] = attn.sigma_signal.detach().cpu().tolist()   # [H] → list
info["sigma_noise"]  = attn.sigma_noise.detach().cpu().tolist()    # [H] → list
```

这是唯一需要同步改动的外部文件，改动量 2 行。

------

## 可视化价值

训练完成后你可以画这样一张图：

```
Head 0:  σ_signal=2.1  σ_noise=12.3   → 聚焦 P300 峰值(~250ms)
Head 1:  σ_signal=4.7  σ_noise=8.5    → 中等时间窗口
Head 2:  σ_signal=1.8  σ_noise=15.1   → 极窄信号 + 极宽背景
Head 3:  σ_signal=6.3  σ_noise=9.2    → 关注晚期慢电位
```

这种分化如果真的发生了，论文中只需要一张柱状图就能很有力地论证"多尺度 ERP 时间建模"的有效性——每个 head 自动发现了不同的 ERP 时间成分。

如果分化**没有**发生（所有 head 的 sigma 收敛到相近的值），这也是有价值的实验结果——说明在当前数据集上单一时间尺度就足够了，可以在论文讨论部分提及。