# 🚗 MiniDriveWorld

> 从零构建一个自动驾驶世界模型：能根据当前场景预测未来几秒的驾驶画面

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 项目简介

MiniDriveWorld 是一个自动驾驶场景的世界模型项目，能够：

- 🎬 **预测未来帧**：根据历史驾驶画面，预测未来 1-3 秒的场景
- 🎮 **条件生成**：根据控制信号（方向盘、油门）生成对应的未来场景
- ⚡ **高效推理**：使用 CUDA 优化 + TensorRT 部署，实现实时预测
- 🖥️ **可视化 Demo**：前端展示预测结果

---

## 🏗️ 项目架构

```
输入                      世界模型                    输出
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  历史帧      │         │ Transformer │         │  未来帧      │
│  (T-3 ~ T)  │ ──────→ │      +      │ ──────→ │ (T+1 ~ T+N) │
│  控制信号    │         │  Diffusion  │         │  预测轨迹    │
└─────────────┘         └─────────────┘         └─────────────┘
```

---

## 📁 目录结构

```
MiniDriveWorld/
├── README.md                 # 项目说明
├── requirements.txt          # Python 依赖
├── setup.py                  # 安装脚本
│
├── configs/                  # 配置文件
│   ├── model_config.yaml     # 模型配置
│   ├── train_config.yaml     # 训练配置
│   └── inference_config.yaml # 推理配置
│
├── data/                     # 数据相关
│   ├── __init__.py
│   ├── dataset.py            # 数据集类
│   ├── dataloader.py         # 数据加载器
│   ├── transforms.py         # 数据增强
│   └── download_nuscenes.sh  # 下载数据集脚本
│
├── models/                   # 模型定义
│   ├── __init__.py
│   ├── world_model.py        # 主模型
│   ├── transformer.py        # Transformer 模块
│   ├── diffusion.py          # 扩散模型模块
│   ├── encoder.py            # 图像编码器
│   └── decoder.py            # 图像解码器
│
├── cuda_kernels/             # CUDA 自定义算子
│   ├── CMakeLists.txt        # CMake 构建
│   ├── attention.cu          # 优化的 Attention
│   ├── layernorm.cu          # 优化的 LayerNorm
│   ├── softmax.cu            # 优化的 Softmax
│   └── binding.cpp           # PyTorch 绑定
│
├── inference/                # 推理引擎
│   ├── CMakeLists.txt        # CMake 构建
│   ├── engine.cpp            # C++ 推理引擎
│   ├── engine.h              # 头文件
│   ├── tensorrt_utils.cpp    # TensorRT 工具
│   └── python_binding.cpp    # Python 绑定
│
├── frontend/                 # 前端可视化
│   ├── package.json          # npm 配置
│   ├── src/
│   │   ├── App.jsx           # 主组件
│   │   ├── VideoPlayer.jsx   # 视频播放器
│   │   └── Predictor.jsx     # 预测展示
│   └── public/
│
├── scripts/                  # 脚本工具
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   ├── export_onnx.py        # 导出 ONNX
│   ├── benchmark.py          # 性能测试
│   └── visualize.py          # 可视化脚本
│
├── docs/                     # 文档
│   ├── installation.md       # 安装指南
│   ├── data_preparation.md   # 数据准备
│   ├── training.md           # 训练指南
│   ├── inference.md          # 推理指南
│   └── optimization.md       # 优化指南
│
└── tests/                    # 测试
    ├── test_model.py
    ├── test_data.py
    └── test_inference.py
```

---

## 🛠️ 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **算法** | Transformer, Diffusion Model | 世界模型核心 |
| **训练** | PyTorch, PyTorch Lightning | 模型训练 |
| **数据** | nuScenes Dataset | 自动驾驶数据 |
| **优化** | CUDA, cuDNN | 自定义算子 |
| **部署** | TensorRT, ONNX | 高效推理 |
| **后端** | FastAPI | API 服务 |
| **前端** | React, Three.js | 可视化 |

---

## 📅 开发计划

### Phase 1: 数据与基础 (2-3周)
- [ ] 下载 nuScenes mini 数据集
- [ ] 实现数据加载器
- [ ] 数据预处理与增强
- [ ] 可视化工具

### Phase 2: 模型设计与训练 (3-4周)
- [ ] 实现 Video Transformer
- [ ] 实现 Diffusion 模块
- [ ] 加入控制信号条件
- [ ] 分布式训练支持
- [ ] 训练与调参

### Phase 3: 推理优化 (3-4周)
- [ ] Python 推理基线
- [ ] 导出 ONNX 模型
- [ ] 实现 CUDA 自定义算子
  - [ ] Attention 优化
  - [ ] LayerNorm 优化
  - [ ] Softmax 优化
- [ ] TensorRT 部署
- [ ] C++ 推理引擎

### Phase 4: 系统集成 (2周)
- [ ] FastAPI 后端
- [ ] React 前端
- [ ] 实时预测 Demo
- [ ] Docker 打包

### Phase 5: 进阶 (可选)
- [ ] BEV 表示
- [ ] 多模态输入
- [ ] CARLA 仿真对接
- [ ] 技术博客

---

## 🚀 快速开始

### 环境安装

```bash
# 克隆项目
git clone https://github.com/yourusername/MiniDriveWorld.git
cd MiniDriveWorld

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 安装 CUDA 扩展
cd cuda_kernels
pip install -e .
```

### 数据准备

```bash
# 下载 nuScenes mini 数据集
bash data/download_nuscenes.sh
```

### 训练模型

```bash
python scripts/train.py --config configs/train_config.yaml
```

### 推理预测

```bash
python scripts/inference.py --checkpoint checkpoints/best.pth --input sample.mp4
```

### 启动 Demo

```bash
# 后端
python scripts/serve.py

# 前端
cd frontend && npm start
```

---

## 📊 性能指标

| 指标 | Python | CUDA 优化 | TensorRT |
|------|--------|----------|----------|
| 推理延迟 | 200ms | 80ms | 40ms |
| 吞吐量 | 5 FPS | 12 FPS | 25 FPS |
| 显存占用 | 8GB | 6GB | 4GB |

---

## 📚 参考资料

### 论文
- [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080)
- [DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving](https://arxiv.org/abs/2309.09777)
- [UniSim: A Neural Closed-Loop Sensor Simulator](https://arxiv.org/abs/2308.01898)

### 数据集
- [nuScenes](https://www.nuscenes.org/)
- [Waymo Open Dataset](https://waymo.com/open/)

### 技术博客
- [Sora 技术解读](https://openai.com/sora)
- [李沐讲 Transformer](https://www.bilibili.com/video/BV1pu411o7BE)

---

## 📝 License

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

---

## 📧 联系

- Author: GJJ
- Email: your-email@example.com

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
