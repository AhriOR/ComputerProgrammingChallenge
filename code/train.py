#!/usr/bin/env python3
"""
花卉分类 ViT-MoE 模型训练脚本
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import tqdm

# 导入自定义模块（保持你的原有数据和模型逻辑）
from code.utils import train_dataloader, val_dataloader
from model import vit_base_patch16_224_moe


# ---------------------- 辅助类与函数（参考示例风格，规范化统计） ----------------------
class AverageMeter:
    """用于统计平均损失和准确率的工具类"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # 当前批次值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 样本数

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, labels, topk=(1,)):
    """计算Top-k准确率（参考示例，支持Top1/Top5）"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        # 获取前maxk个预测结果的索引
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()  # 转置为 (k, batch_size)
        correct = pred.eq(labels.view(1, -1).expand_as(pred))  # 与真实标签对比

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))  # 转为百分比
        return res


def save_config(config, save_path):
    """保存训练配置到JSON文件（参考示例的配置保存逻辑）"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"✅ 训练配置已保存至: {save_path}")


def set_seed(seed=42):
    """设置随机种子，保证实验可复现（参考示例）"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------- 核心训练/验证函数（拆分逻辑，参考示例结构） ----------------------
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """单轮训练函数（规范化统计与日志输出）"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()  # Top1准确率

    train_loop = tqdm.tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch [{epoch+1}]/Train",
        leave=False
    )

    for batch_idx, (images, labels) in train_loop:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计指标
        acc1 = calculate_accuracy(outputs, labels, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        # 更新进度条
        train_loop.set_postfix({
            "batch_loss": f"{loss.item():.4f}",
            "avg_loss": f"{losses.avg:.4f}",
            "avg_acc1": f"{top1.avg:.2f}%"
        })

    # 打印单轮训练总结
    print(f"Epoch [{epoch+1}]/Train | Loss: {losses.avg:.4f} | Acc@1: {top1.avg:.2f}%")
    return losses.avg, top1.avg


def validate_epoch(model, val_loader, criterion, device):
    """单轮验证函数（无梯度计算，规范化统计）"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():  # 关闭梯度，节省显存
        val_loop = tqdm.tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc="Validating",
            leave=False
        )

        for batch_idx, (images, labels) in val_loop:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 统计指标
            acc1 = calculate_accuracy(outputs, labels, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # 更新进度条
            val_loop.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{losses.avg:.4f}",
                "avg_acc1": f"{top1.avg:.2f}%"
            })

    # 打印单轮验证总结
    print(f"Validation | Loss: {losses.avg:.4f} | Acc@1: {top1.avg:.2f}%")
    return losses.avg, top1.avg


# ---------------------- 主函数（命令行参数、训练流程控制，参考示例核心结构） ----------------------
def main():
    # 1. 解析命令行参数（参考示例，支持灵活配置）
    parser = argparse.ArgumentParser(description='花卉分类 ViT-MoE 模型训练')
    # 模型参数
    parser.add_argument('--model_type', type=str, default='vit_moe', help='模型类型（固定为ViT-MoE）')
    parser.add_argument('--num_classes', type=int, default=100, help='类别数量')
    parser.add_argument('--num_experts', type=int, default=4, help='MoE专家数量')
    parser.add_argument('--top_k', type=int, default=2, help='MoE每样本选择专家数')
    parser.add_argument('--embed_dim', type=int, default=512, help='ViT嵌入维度')
    parser.add_argument('--depth', type=int, default=6, help='ViT Transformer块数量')
    parser.add_argument('--num_heads', type=int, default=8, help='多头注意力头数')
    parser.add_argument('--mlp_ratio', type=float, default=2.0, help='MLP隐藏层比例')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像尺寸')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch分块尺寸')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-4, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup轮数')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='余弦退火最小学习率')
    # 路径与设备参数
    parser.add_argument('--save_dir', type=str, default='../models', help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的Checkpoint路径')
    parser.add_argument('--device', type=str, default=None, help='指定设备（cuda/cpu，默认自动检测）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 2. 基础配置初始化
    set_seed(args.seed)  # 固定种子
    # 自动检测设备（优先cuda）
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    print(f"📌 使用设备: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"📌 模型保存目录: {args.save_dir}")

    # 3. 数据加载（保留你的原有逻辑，从code.utils导入）
    print("\n🔍 加载数据加载器...")
    train_loader = train_dataloader
    val_loader = val_dataloader
    # 获取类别映射（从数据集的LabelEncoder获取）
    # 注：train_loader.dataset 是Subset，其dataset是PlantImageDataset实例
    label_encoder = train_loader.dataset.dataset.label_encoder
    class_to_idx = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print(f"✅ 数据加载完成 | 训练集样本: {len(train_loader.dataset)} | 验证集样本: {len(val_loader.dataset)} | 类别数: {len(class_to_idx)}")

    # 4. 模型初始化（保留你的ViT-MoE实例化逻辑）
    print("\n🔧 创建 ViT-MoE 模型...")
    model = vit_base_patch16_224_moe(
        num_experts=args.num_experts,
        top_k=args.top_k,
        classes=args.num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channel=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio
    ).to(device)
    print(f"✅ 模型创建完成 | 模型参数总量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # 5. 优化器与学习率调度器（保留你的Warmup+余弦退火逻辑）
    print("\n🔧 初始化优化器与学习率调度器...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-08
    )
    # Warmup调度器
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / args.warmup_epochs)
    # 余弦退火调度器（总轮数=总epoch - warmup轮数）
    main_epochs = args.epochs - args.warmup_epochs
    main_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=args.eta_min)
    # 组合调度器
    lr_scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [args.warmup_epochs])
    print(f"✅ 优化器: AdamW | 初始LR: {args.lr} | Warmup轮数: {args.warmup_epochs} | 调度器: Warmup+CosineAnnealing")

    # 6. 损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # 7. 恢复训练（参考示例的Checkpoint加载逻辑）
    start_epoch = 0
    best_val_acc = 0.0
    # 训练历史记录（用于后续分析）
    history = {
        'train_loss': [],
        'train_acc1': [],
        'val_loss': [],
        'val_acc1': []
    }

    if args.resume is not None and os.path.exists(args.resume):
        print(f"\n🔄 从Checkpoint恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        # 加载参数
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        history = checkpoint.get('history', history)
        print(f"✅ 恢复完成 | 起始Epoch: {start_epoch} | 历史最佳Val Acc: {best_val_acc:.2f}%")
    elif args.resume is not None:
        print(f"⚠️ 未找到Checkpoint: {args.resume}，将从头开始训练")

    # 8. 核心训练循环（参考示例的流程，整合训练/验证/保存）
    print("\n🚀 开始训练...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # 8.1 单轮训练
        train_loss, train_acc1 = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        # 8.2 单轮验证
        val_loss, val_acc1 = validate_epoch(model, val_loader, criterion, device)
        # 8.3 更新学习率调度器
        lr_scheduler.step()
        # 8.4 记录训练历史
        history['train_loss'].append(train_loss)
        history['train_acc1'].append(train_acc1)
        history['val_loss'].append(val_loss)
        history['val_acc1'].append(val_acc1)

        # 8.5 计算epoch耗时
        epoch_time = time.time() - epoch_start_time
        print(f"⏱️ Epoch [{epoch+1}] 耗时: {epoch_time:.1f}s | 当前LR: {optimizer.param_groups[0]['lr']:.6f}\n")

        # 8.6 保存Checkpoint（参考示例的双Checkpoint策略：latest + best）
        # 保存最新Checkpoint（每次epoch都保存）
        latest_ckpt_path = os.path.join(args.save_dir, 'latest_checkpoint.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history,
            'args': vars(args)  # 保存命令行参数
        }, latest_ckpt_path)

        # 保存最佳Checkpoint（仅当验证准确率提升时）
        is_best = val_acc1 > best_val_acc
        if is_best:
            best_val_acc = val_acc1
            best_ckpt_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'args': vars(args)
            }, best_ckpt_path)
            print(f"🌟 最佳模型已更新 | 新最佳Val Acc: {best_val_acc:.2f}% | 保存路径: {best_ckpt_path}\n")

    # 9. 训练完成后：保存配置与历史记录（参考示例的收尾逻辑）
    print("\n🏁 训练全部完成！")
    print(f"📊 最终最佳验证准确率: {best_val_acc:.2f}%")

    # 保存训练配置
    config = {
        'model_config': {
            'model_type': args.model_type,
            'num_classes': args.num_classes,
            'num_experts': args.num_experts,
            'embed_dim': args.embed_dim,
            'img_size': args.img_size
        },
        'training_params': vars(args),
        'class_to_idx': class_to_idx,
        'best_val_acc': best_val_acc,
        'training_history': history
    }
    save_config(config, os.path.join(args.save_dir, 'train_config.json'))

    # 保存训练历史（可选，方便后续分析）
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
    print(f"✅ 训练历史已保存至: {os.path.join(args.save_dir, 'training_history.json')}")


if __name__ == "__main__":
    main()