#!/usr/bin/env python3
"""
MiniDriveWorld 训练脚本

用法:
    python scripts/train.py --config configs/train_config.yaml
    
    # 多卡训练
    torchrun --nproc_per_node=4 scripts/train.py --config configs/train_config.yaml
"""

import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# TODO: 取消注释
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning.loggers import TensorBoardLogger

# 本地模块
import sys
sys.path.append(str(Path(__file__).parent.parent))

# from data import NuScenesDataset, create_dataloader
# from models import MiniDriveWorldModel


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config: dict):
    """训练主函数"""
    print("=" * 50)
    print("MiniDriveWorld Training")
    print("=" * 50)
    
    # TODO: 实现训练逻辑
    
    # 1. 创建数据集
    print("\n[1/5] 创建数据集...")
    # train_dataset = NuScenesDataset(...)
    # val_dataset = NuScenesDataset(...)
    # train_loader = create_dataloader(train_dataset, ...)
    # val_loader = create_dataloader(val_dataset, ...)
    
    # 2. 创建模型
    print("[2/5] 创建模型...")
    # model = MiniDriveWorldModel(config['model'])
    
    # 3. 创建优化器
    print("[3/5] 创建优化器...")
    # optimizer = AdamW(model.parameters(), **config['training']['optimizer'])
    # scheduler = CosineAnnealingLR(optimizer, **config['training']['scheduler'])
    
    # 4. 训练循环
    print("[4/5] 开始训练...")
    # for epoch in range(config['training']['epochs']):
    #     train_one_epoch(model, train_loader, optimizer, epoch)
    #     validate(model, val_loader, epoch)
    #     scheduler.step()
    
    # 5. 保存模型
    print("[5/5] 保存模型...")
    # torch.save(model.state_dict(), 'checkpoints/final.pth')
    
    print("\n训练完成！")


def train_one_epoch(model, dataloader, optimizer, epoch):
    """训练一个 epoch"""
    model.train()
    
    for batch_idx, batch in enumerate(dataloader):
        # 前向传播
        output = model(
            input_frames=batch['input_frames'],
            control_signals=batch['control_signals'],
            target_frames=batch['target_frames'],
        )
        
        loss = output['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 日志
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")


def validate(model, dataloader, epoch):
    """验证"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            output = model(
                input_frames=batch['input_frames'],
                control_signals=batch['control_signals'],
                target_frames=batch['target_frames'],
            )
            total_loss += output['loss'].item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} | Validation Loss: {avg_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='MiniDriveWorld Training')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 开始训练
    train(config)


if __name__ == '__main__':
    main()
