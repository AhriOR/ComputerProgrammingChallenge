import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

# 全局字典：存储类别映射（原始ID -> 中文名称）
image_category_dict = {}


class PlantImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        植物图像数据集（优化版）：自动过滤无效图片，确保数据加载稳定
        Args:
            csv_file: CSV文件路径（包含filename、category_id等字段）
            image_dir: 图像存放目录
            transform: 图像预处理管道
        """
        self.image_dir = image_dir
        self.transform = transform

        # 1. 读取CSV文件（支持多种编码）
        try:
            self.data_frame = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.data_frame = pd.read_csv(csv_file, encoding='gbk')
            except:
                self.data_frame = pd.read_csv(csv_file, encoding='latin-1')

        # 2. 过滤无效图片路径（核心优化：提前排除不存在的图片）
        valid_rows = []
        invalid_count = 0  # 统计无效样本数量
        for idx, row in self.data_frame.iterrows():
            img_path = os.path.join(self.image_dir, row['filename'])
            # 检查文件是否存在且非空（避免0字节的损坏文件）
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                valid_rows.append(row)
            else:
                invalid_count += 1
                # 打印无效样本信息（方便后续核对）
                if invalid_count <= 10:  # 只打印前10个无效样本，避免日志冗余
                    print(f"⚠️ 过滤无效图片（不存在或为空）：{img_path}")

        # 更新数据框为仅包含有效样本
        self.data_frame = pd.DataFrame(valid_rows)
        print(
            f"\n数据过滤完成：原始样本数 {len(valid_rows) + invalid_count}，有效样本数 {len(self.data_frame)}，过滤无效样本 {invalid_count}")

        # 3. 标签编码（基于过滤后的有效样本）
        self.label_encoder = LabelEncoder()
        self.data_frame['encoded_label'] = self.label_encoder.fit_transform(self.data_frame['category_id'])

        # 4. 打印数据集基本信息
        print(f"有效数据集大小: {len(self.data_frame)}")
        print(f"类别数量: {self.data_frame['encoded_label'].nunique()}")
        print(f"标签范围: {self.data_frame['encoded_label'].min()} - {self.data_frame['encoded_label'].max()}")
        print(f"类别分布（前5类）:\n{self.data_frame['encoded_label'].value_counts().head()}")

        # 5. 打印类别映射示例
        print("\n类别映射示例:")
        sample_mapping = self.data_frame[['category_id', 'encoded_label', 'chinese_name']].drop_duplicates().head(10)
        for _, row in sample_mapping.iterrows():
            print(f"  原始ID: {row['category_id']} -> 编码后: {row['encoded_label']}（{row['chinese_name']}）")
            # 更新全局类别映射字典
            image_category_dict[str(row['category_id'])] = row['chinese_name']

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 获取样本信息
        row = self.data_frame.iloc[idx]
        filename = row['filename']
        category_id = row['category_id']
        encoded_label = row['encoded_label']

        # 构建图像路径（已提前过滤，理论上不会不存在）
        img_path = os.path.join(self.image_dir, filename)

        # 加载图像（处理可能的损坏文件）
        try:
            image = Image.open(img_path).convert('RGB')
            # 额外检查图像是否损坏（部分文件存在但无法解析）
            if image.size == (0, 0):
                raise ValueError("图像尺寸为0，可能损坏")
        except Exception as e:
            print(f"❌ 加载图片失败（损坏）: {img_path}, 错误: {e}")
            # 创建黑色图像作为替代（避免训练中断）
            image = Image.new('RGB', (224, 224), color='black')

        # 应用预处理
        if self.transform:
            image = self.transform(image)

        return image, encoded_label

    def get_label_mapping(self):
        """获取原始类别ID到编码标签的映射"""
        return dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))


# 数据预处理管道（增强鲁棒性）
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 先放大到256，再裁剪，保留更多细节
    transforms.RandomCrop(224),  # 随机裁剪224x224
    transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
    transforms.RandomVerticalFlip(p=0.2),  # 20%概率垂直翻转（新增，增强多样性）
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet均值
        std=[0.229, 0.224, 0.225]  # ImageNet标准差
    )
])

# 创建数据集
dataset = PlantImageDataset(
    csv_file=r"F:\计挑赛\train_labels.csv",
    image_dir=r"F:\计挑赛\train",
    transform=transform
)

# 获取标签映射（原始ID -> 编码标签）
label_mapping = dataset.get_label_mapping()
print(f"\n总类别数量: {len(label_mapping)}")

# 创建数据加载器（优化稳定性）
train_dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    drop_last=True  # 丢弃最后一个不完整的批次，避免训练时的形状不匹配
)


