"""
NuScenes 数据集加载器

TODO: 实现以下功能
1. 加载 nuScenes 数据集
2. 提取连续帧序列
3. 提取控制信号（方向盘、油门等）
4. 数据预处理
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class NuScenesDataset(Dataset):
    """
    nuScenes 数据集
    
    用于加载自动驾驶场景的连续帧序列
    
    Args:
        data_root: 数据集根目录
        version: 数据集版本 (v1.0-mini, v1.0-trainval)
        split: 数据划分 (train, val, test)
        num_input_frames: 输入历史帧数
        num_output_frames: 预测未来帧数
        transform: 数据变换
    """
    
    def __init__(
        self,
        data_root: str,
        version: str = "v1.0-mini",
        split: str = "train",
        num_input_frames: int = 4,
        num_output_frames: int = 8,
        transform = None,
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.version = version
        self.split = split
        self.num_input_frames = num_input_frames
        self.num_output_frames = num_output_frames
        self.transform = transform
        
        # TODO: 初始化 nuScenes API
        # from nuscenes.nuscenes import NuScenes
        # self.nusc = NuScenes(version=version, dataroot=data_root)
        
        # TODO: 构建样本索引
        self.samples = self._build_samples()
        
    def _build_samples(self) -> List[Dict]:
        """
        构建数据样本索引
        
        每个样本包含:
        - scene_token: 场景标识
        - start_frame: 起始帧索引
        - frame_tokens: 帧 token 列表
        """
        samples = []
        
        # TODO: 遍历所有场景，提取连续帧序列
        # for scene in self.nusc.scene:
        #     ...
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            dict: {
                'input_frames': [T_in, C, H, W] 输入帧
                'target_frames': [T_out, C, H, W] 目标帧
                'control_signals': [T_in + T_out, D] 控制信号
                'timestamps': [T_in + T_out] 时间戳
            }
        """
        sample = self.samples[idx]
        
        # TODO: 加载图像帧
        input_frames = self._load_frames(sample, is_input=True)
        target_frames = self._load_frames(sample, is_input=False)
        
        # TODO: 加载控制信号
        control_signals = self._load_control_signals(sample)
        
        # 应用数据变换
        if self.transform:
            input_frames = self.transform(input_frames)
            target_frames = self.transform(target_frames)
        
        return {
            'input_frames': input_frames,
            'target_frames': target_frames,
            'control_signals': control_signals,
        }
    
    def _load_frames(self, sample: Dict, is_input: bool) -> torch.Tensor:
        """加载图像帧"""
        # TODO: 实现图像加载
        pass
    
    def _load_control_signals(self, sample: Dict) -> torch.Tensor:
        """加载控制信号（方向盘、油门、刹车、速度）"""
        # TODO: 实现控制信号加载
        pass


# 测试代码
if __name__ == "__main__":
    # 测试数据集
    dataset = NuScenesDataset(
        data_root="./data/nuscenes",
        version="v1.0-mini",
        split="train",
    )
    print(f"数据集大小: {len(dataset)}")
