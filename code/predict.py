#!/usr/bin/env python3
"""
花卉分类 ViT-MoE 模型预测脚本

使用方法:
    python ./code/predict.py <测试集CSV路径> <测试集图片目录> <输出文件路径>

示例:
    python ./code/predict.py ./data/test_labels.csv ./data/test ./results/submission.csv

输出格式:
    CSV文件包含三列: filename, category_id, confidence
    - filename: 测试图片文件名（与测试集CSV一致）
    - category_id: 预测的花卉类别ID（原始类别编号）
    - confidence: 预测置信度（0-1之间，保留4位小数）
"""

import os
import argparse
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import joblib


# ---------------------- 1. 测试集数据集类（保留原有核心逻辑） ----------------------
class TestPlantImageDataset(Dataset):
    def __init__(self, test_csv_path, test_image_dir, transform=None):
        self.test_image_dir = test_image_dir
        self.transform = transform

        # 读取测试集CSV（支持多编码）
        try:
            self.test_df = pd.read_csv(test_csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.test_df = pd.read_csv(test_csv_path, encoding='gbk')

        # 校验CSV格式（必须包含"image_name"列）
        assert "image_name" in self.test_df.columns, "测试集CSV必须包含'image_name'列"
        self.image_names = self.test_df["image_name"].tolist()  # 从CSV获取图片名

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.test_image_dir, image_name)

        # 加载图片（处理损坏/异常文件）
        try:
            image = Image.open(image_path).convert('RGB')
            if image.size == (0, 0):
                raise ValueError("图片尺寸为空")
        except Exception as e:
            print(f"⚠️ 加载图片失败: {image_path} | 错误: {e} | 用黑色图替代")
            image = Image.new('RGB', (600, 600), color='black')  # 赛题图片默认600×600

        # 应用预处理（无随机操作，保证推理一致）
        if self.transform:
            image = self.transform(image)

        return image, image_name  # 返回：处理后图片 + 原始文件名


# ---------------------- 2. 核心预测函数（保留真实模型推理逻辑） ----------------------
def model_predict(model, test_loader, device, model_path, label_encoder_path, output_csv_path):
    """
    用训练好的ViT-MoE模型预测测试集，生成标准CSV结果
    """
    # 加载最优模型权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重文件不存在: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 加载模型成功 | 训练最佳验证准确率: {checkpoint.get('best_val_acc', 0):.2f}% | 训练轮次: {checkpoint.get('epoch', 0)}")

    # 加载LabelEncoder（编码标签→原始类别ID）
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"LabelEncoder文件不存在: {label_encoder_path}")
    label_encoder = joblib.load(label_encoder_path)
    print(f"✅ 加载LabelEncoder成功 | 类别总数: {len(label_encoder.classes_)}")

    # 模型切换为评估模式（关闭Dropout/MoE随机选择）
    model.eval()
    predictions = []  # 存储结果：(filename, category_id, confidence)

    # 无梯度推理（节省显存+加速）
    with torch.no_grad():
        predict_loop = tqdm.tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc="🔍 测试集预测中",
            leave=True
        )

        for batch_idx, (images, image_names) in predict_loop:
            images = images.to(device)

            # 模型前向传播（输出logits）
            outputs = model(images)  # shape: (batch_size, num_classes)
            # 计算置信度（softmax归一化→概率分布）
            probs = torch.softmax(outputs, dim=1)
            # 获取每个样本的最大置信度及对应类别索引
            max_probs, encoded_preds = torch.max(probs, dim=1)

            # 编码标签→原始类别ID（与赛题类别ID一致）
            raw_category_ids = label_encoder.inverse_transform(encoded_preds.cpu().numpy())
            # 置信度转为Python数值（保留4位小数）
            confidences = [round(prob.item(), 4) for prob in max_probs]

            # 收集当前批次结果
            batch_results = list(zip(image_names, raw_category_ids, confidences))
            predictions.extend(batch_results)

    # 生成标准CSV（列名与参考脚本一致：filename, category_id, confidence）
    pred_df = pd.DataFrame(
        predictions,
        columns=["filename", "category_id", "confidence"]
    )
    # 按文件名排序（保证与测试集顺序一致）
    pred_df = pred_df.sort_values("filename").reset_index(drop=True)

    # 保存CSV（UTF-8编码，避免服务器解析乱码）
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")
    pred_df.to_csv(output_csv_path, index=False, encoding='utf-8')

    # 打印预测总结
    print(f"\n🎉 预测完成！")
    print(f"📊 预测样本总数: {len(pred_df)}")
    print(f"💾 结果保存路径: {output_csv_path}")
    print(f"📄 CSV格式示例:\n{pred_df.head(3)}")

    return pred_df


# ---------------------- 3. 主函数（对齐参考脚本的命令行参数风格） ----------------------
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='花卉分类 ViT-MoE 模型预测脚本')

    # 位置参数（必须传入，顺序固定）
    parser.add_argument('test_csv_path', type=str,
                        help='测试集CSV文件路径（需包含"image_name"列）')
    parser.add_argument('test_img_dir', type=str,
                        help='测试集图片存放目录')
    parser.add_argument('output_path', type=str,
                        help='预测结果输出CSV路径（如./results/submission.csv）')

    # 可选参数（默认值对齐你的原有配置）
    parser.add_argument('--model_path', type=str, default='./models/best_model.pth',
                        help='训练好的模型权重路径（默认: ./models/best_model.pth）')
    parser.add_argument('--label_encoder_path', type=str, default='./models/label_encoder.pkl',
                        help='LabelEncoder文件路径（默认: ./models/label_encoder.pkl）')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='推理批次大小（默认: 32，根据显存调整）')
    parser.add_argument('--img_size', type=int, default=224,
                        help='模型输入图像尺寸（默认: 224，需与训练一致）')
    parser.add_argument('--device', type=str, default=None,
                        help='指定运行设备（cuda/cpu，默认自动检测）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（保证损坏图片处理一致性，默认: 42）')

    args = parser.parse_args()

    # 基础配置初始化
    torch.manual_seed(args.seed)  # 固定种子，确保可复现
    # 自动检测设备（优先cuda）
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)

    # 打印参数信息（对齐参考脚本的日志风格）
    print("=" * 60)
    print("📋 预测参数配置")
    print("=" * 60)
    print(f"测试集CSV路径: {args.test_csv_path}")
    print(f"测试集图片目录: {args.test_img_dir}")
    print(f"结果输出路径: {args.output_path}")
    print(f"模型权重路径: {args.model_path}")
    print(f"LabelEncoder路径: {args.label_encoder_path}")
    print(f"批次大小: {args.batch_size} | 输入尺寸: {args.img_size} | 设备: {args.device} | 种子: {args.seed}")
    print("=" * 60)

    # 检查必要路径是否存在
    if not os.path.exists(args.test_csv_path):
        print(f"❌ 错误: 测试集CSV不存在: {args.test_csv_path}")
        return
    if not os.path.exists(args.test_img_dir):
        print(f"❌ 错误: 测试集图片目录不存在: {args.test_img_dir}")
        return

    # 1. 初始化测试集预处理（与训练一致，无随机增强）
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 先缩放至256×256
        transforms.CenterCrop(args.img_size),  # 中心裁剪至输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(  # ImageNet均值/标准差（与训练一致）
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. 初始化测试集加载器
    test_dataset = TestPlantImageDataset(
        test_csv_path=args.test_csv_path,
        test_image_dir=args.test_img_dir,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 测试集不打乱，保证顺序一致
        pin_memory=True,  # 加速数据传输到GPU
        num_workers=2  # Windows建议≤2，避免线程报错
    )
    print(f"✅ 测试集加载完成 | 图片总数: {len(test_dataset)} | 批次总数: {len(test_loader)}")

    # 3. 初始化ViT-MoE模型（参数与训练一致）
    from code.model import vit_base_patch16_224_moe  # 导入你的模型
    model = vit_base_patch16_224_moe(
        num_experts=2,
        top_k=1,
        classes=100,  # 赛题100类花卉
        img_size=args.img_size,
        patch_size=16,
        in_channel=3,
        embed_dim=256,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0
    ).to(device)
    print(f"✅ ViT-MoE模型初始化完成 | 设备: {args.device}")

    # 4. 启动模型预测
    model_predict(
        model=model,
        test_loader=test_loader,
        device=device,
        model_path=args.model_path,
        label_encoder_path=args.label_encoder_path,
        output_csv_path=args.output_path
    )


if __name__ == '__main__':
    main()