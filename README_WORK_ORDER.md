# 🧭 MiniDriveWorld 开工顺序与先修指南

> 这份文档是给“现在时间碎片化，但想后续顺畅开工”的你准备的。  
> 目标：先做铺垫，后续有时间时可以直接进入高产状态。

---

## 0. 先回答一个核心问题：现在该做什么？

你当前最优策略不是马上啃完所有模块，而是：

1. 先把 **最小可跑通链路** 搭起来  
2. 用最小数据做 **端到端 smoke test**  
3. 再逐步替换成真实实现（dataset / model / train / kernel）

一句话：**先通路，再提精度，再做优化。**

---

## 1. 保底先修：学完这些就能开工

以下是“够用就行”的最低门槛，不需要一开始就学很深。

### 1.1 Python / PyTorch（必须）

- 会写 `Dataset` / `DataLoader`
- 会写 `nn.Module`、`forward`
- 会写基础训练循环（loss、backward、optimizer.step）
- 会看 tensor shape（这是世界模型最关键能力）

### 1.2 机器学习基础（必须）

- Transformer 基本结构（Attention、残差、层归一化）
- Diffusion 的高层概念（噪声预测，不要求先抠公式）
- 训练/验证拆分、过拟合与欠拟合概念

### 1.3 CUDA（建议但非开工阻塞）

- 先到“会写基础 kernel + 会 profile”即可
- 现在 CUDA 可先作为后期加分项，不阻塞主线

---

## 2. 工作顺序（按文件级）

## Phase A：先跑通最小主链路（最优先）

先改这 3 个文件：

1. `models/world_model.py`  
2. `data/dataset.py`  
3. `scripts/train.py`

### A1 `models/world_model.py`

先做一个极简可训练版本（哪怕性能差）：

- `encoder`：简单 CNN 或小 backbone
- `transformer`：先用 1~2 层
- `diffusion`：先用简化占位 loss（后续再换）
- 保证 `forward` 输出结构稳定

验收标准：

- 输入 `[B, T_in, C, H, W]`，能稳定输出预测张量和 loss

### A2 `data/dataset.py`

先不追求完整 nuScenes 特性，先做可用：

- `_build_samples` 返回可用样本索引
- `_load_frames` 返回固定尺寸张量
- `_load_control_signals` 返回对齐的控制信号

验收标准：

- `len(dataset) > 0`
- `__getitem__` 能返回 `input_frames/target_frames/control_signals`

### A3 `scripts/train.py`

先跑“微型训练”：

- 小 batch、小分辨率、1~2 epoch
- 打印 loss，不追求指标
- 每 N step 打印 shape + loss

验收标准：

- 能完整跑完一个 epoch，无报错

---

## Phase B：把“能跑”升级到“有结果”

优先改：

- `configs/model_config.yaml`
- `configs/train_config.yaml`
- `scripts/train.py`（日志、checkpoint、验证）

要做：

- 配置参数可控（input/output frame 数、hidden dim、lr）
- 保存 checkpoint
- 加最基本可视化输出（预测帧样例）

验收标准：

- 有可复现实验配置
- 有可视化结果（哪怕质量一般）

---

## Phase C：再做优化（CUDA / TensorRT）

优先改：

- `cuda_kernels/attention.cu`
- 后续 `layernorm.cu`、`softmax.cu`

要做：

- 先实现 `naive`（可验证）
- 再实现 `tiled`（可比较）
- 最后考虑 flash-style

验收标准：

- 与 PyTorch baseline 对齐（数值误差可控）
- 至少有一次速度对比报告

---

## 3. 你的“保底目标”与“进阶目标”

### 保底目标（可写简历）

- 数据能读
- 模型能训
- 能输出预测结果
- 有一份实验记录

### 进阶目标（高含金量）

- 有 CUDA 优化对比
- 有推理延迟优化数据（ONNX/TensorRT）
- 有 demo 可视化

---

## 4. 推荐学习资料（够用优先）

### PyTorch / 训练工程

- PyTorch 官方教程（Dataset、Training Loop）
- Lightning 文档（可选，用于提效）

### 世界模型 / 自动驾驶

- GAIA-1
- DriveDreamer
- UniSim

### Transformer / Diffusion

- Attention Is All You Need
- DDPM（先读高层概念）

### CUDA 优化

- CUDA C++ Programming Guide
- Nsight Compute 文档
- FlashAttention 论文（做优化阶段再深读）

---

## 5. 一周开工模板（可直接执行）

Day 1-2：

- 跑通 `dataset + world_model` 的 shape 检查

Day 3-4：

- 跑通最小训练循环（1 epoch）

Day 5：

- 输出第一版预测样例

Day 6：

- 修 bug + 记录问题清单

Day 7：

- 写一页周报（做了什么、卡点、下周计划）

---

## 6. 现在就做的第一件事（10 分钟）

执行顺序：

1. 打开 `data/dataset.py`，把 `_build_samples` 改到返回非空样本  
2. 打开 `models/world_model.py`，让 `forward` 返回可训练 loss  
3. 打开 `scripts/train.py`，先跑一个极小配置

只要这三步打通，项目就从“骨架”进入“可迭代”状态。

