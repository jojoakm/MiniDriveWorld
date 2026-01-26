# MiniDriveWorld 学习路线图

> 这是一份详细的学习路线，帮助你从零开始完成这个项目

---

## 🎯 项目目标

构建一个自动驾驶世界模型，能够：
- 根据历史帧预测未来驾驶场景
- 根据控制信号（方向盘、油门）生成对应画面
- 实现高效的推理（CUDA 优化）

---

## 📅 详细计划

### Phase 1: 基础知识准备（2-3周）

#### Week 1: C++ 与系统编程
- [ ] 完成线程学习（pthread / std::thread）
- [ ] 完成多线程实践（锁、同步）
- [ ] 完成 Makefile 学习
- [ ] 练习：手写一个简单的线程池

#### Week 2: CUDA 入门
- [ ] 学习 GPU 架构（SM, Warp, Thread）
- [ ] 学习 CUDA 编程模型
- [ ] 写第一个 CUDA Kernel（向量加法）
- [ ] 学习 Shared Memory 优化
- [ ] 练习：实现矩阵乘法（Naive + Tiled）

#### Week 3: 深度学习基础
- [ ] 复习 PyTorch 基础
- [ ] 学习 Transformer 架构
- [ ] 学习扩散模型基础
- [ ] 阅读论文：GAIA-1, DriveDreamer

---

### Phase 2: 数据与模型（3-4周）

#### Week 4: 数据处理
- [ ] 下载 nuScenes mini 数据集
- [ ] 学习 nuScenes 数据格式
- [ ] 实现 `data/dataset.py`
- [ ] 实现数据增强 `data/transforms.py`
- [ ] 可视化数据样本

#### Week 5-6: 模型实现
- [ ] 实现图像编码器 `models/encoder.py`
- [ ] 实现 Video Transformer `models/transformer.py`
- [ ] 实现扩散模块 `models/diffusion.py`
- [ ] 实现图像解码器 `models/decoder.py`
- [ ] 组装完整模型 `models/world_model.py`

#### Week 7: 训练
- [ ] 完善训练脚本 `scripts/train.py`
- [ ] 在 mini 数据集上训练
- [ ] 调参优化
- [ ] 可视化训练结果

---

### Phase 3: 推理优化（3-4周）

#### Week 8: Python 推理基线
- [ ] 实现推理脚本
- [ ] 测量 Python 推理性能
- [ ] 分析性能瓶颈（Profiling）

#### Week 9: CUDA 优化
- [ ] 实现优化的 Attention Kernel
- [ ] 实现优化的 LayerNorm Kernel
- [ ] 实现优化的 Softmax Kernel
- [ ] 绑定到 PyTorch
- [ ] 性能对比测试

#### Week 10: TensorRT 部署
- [ ] 导出 ONNX 模型
- [ ] TensorRT 转换
- [ ] C++ 推理引擎封装
- [ ] 性能测试与优化

---

### Phase 4: 系统集成（2周）

#### Week 11: 后端开发
- [ ] FastAPI 服务搭建
- [ ] 推理 API 实现
- [ ] 视频流处理

#### Week 12: 前端与部署
- [ ] React 前端开发
- [ ] 可视化组件
- [ ] Docker 打包
- [ ] 完整 Demo 测试

---

## 📚 学习资源

### 视频教程
- [ ] [李沐讲 Transformer](https://www.bilibili.com/video/BV1pu411o7BE)
- [ ] [CUDA 编程入门](https://www.bilibili.com/video/BV1kx411m7Fk)
- [ ] [PyTorch 官方教程](https://pytorch.org/tutorials/)

### 论文阅读
- [ ] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] [DDPM](https://arxiv.org/abs/2006.11239)
- [ ] [GAIA-1](https://arxiv.org/abs/2309.17080)
- [ ] [DriveDreamer](https://arxiv.org/abs/2309.09777)
- [ ] [FlashAttention](https://arxiv.org/abs/2205.14135)

### 代码参考
- [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
- [MILE](https://github.com/wayveai/mile)
- [Diffusers](https://github.com/huggingface/diffusers)

---

## ✅ 检查点

### Phase 1 完成标准
- [ ] 能手写线程池
- [ ] 能写 CUDA 矩阵乘法
- [ ] 理解 Transformer 架构

### Phase 2 完成标准
- [ ] 数据加载器能正常工作
- [ ] 模型能在 mini 数据集上训练
- [ ] 能生成有意义的预测帧

### Phase 3 完成标准
- [ ] CUDA 优化后推理速度提升 3x+
- [ ] TensorRT 部署延迟 < 100ms

### Phase 4 完成标准
- [ ] 完整的 Web Demo 可运行
- [ ] Docker 一键部署

---

## 📝 简历写法

完成后，简历可以这样写：

```
项目：MiniDriveWorld - 自动驾驶世界模型

• 从零实现基于 Transformer + Diffusion 的自动驾驶场景预测模型
• 使用 nuScenes 数据集训练，支持根据控制信号预测未来 3 秒驾驶画面
• 实现 CUDA 自定义算子优化 Attention 计算，推理速度提升 5x
• 使用 TensorRT 部署，端到端延迟 < 50ms
• 开发完整的 Web Demo，支持实时预测展示

技术栈：PyTorch, CUDA, C++, TensorRT, FastAPI, React
```

---

## 🚀 开始吧！

从 Phase 1 Week 1 开始，一步一步来！

有问题随时问 AI 助手！💪
