import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split  # 导入random_split用于划分
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

        # 3. 标签编码（基于过滤后的有效样本，全局统一编码，划分后子集共用）
        self.label_encoder = LabelEncoder()
        self.data_frame['encoded_label'] = self.label_encoder.fit_transform(self.data_frame['category_id'])

        # 4. 打印数据集基本信息
        print(f"有效数据集总大小: {len(self.data_frame)}")
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
        # 获取样本信息（idx为数据集全局索引，划分后子集会自动映射）
        row = self.data_frame.iloc[idx]
        filename = row['filename']
        encoded_label = row['encoded_label']

        # 构建图像路径
        img_path = os.path.join(self.image_dir, filename)

        # 加载图像（处理可能的损坏文件）
        try:
            image = Image.open(img_path).convert('RGB')
            if image.size == (0, 0):
                raise ValueError("图像尺寸为0，可能损坏")
        except Exception as e:
            print(f"❌ 加载图片失败（损坏）: {img_path}, 错误: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        # 应用当前数据集的预处理（训练集带增强，验证集无增强）
        if self.transform:
            image = self.transform(image)

        return image, encoded_label

    def get_label_mapping(self):
        """获取原始类别ID到编码标签的映射（全局统一）"""
        return dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))


# ---------------------- 核心修改：分别定义训练集/验证集预处理 ----------------------
# 1. 训练集预处理（带随机增强，提升泛化能力）
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    # 新增增强
    transforms.RandomRotation(degrees=15),  # 随机旋转±15度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
    transforms.RandomGrayscale(p=0.1),  # 10%概率转为灰度图
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 2. 验证集预处理（无随机操作，确保评估结果稳定）
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),  # 中心裁剪（替代随机裁剪，验证专用）
    transforms.ToTensor(),  # 无翻转操作
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 与训练集归一化一致
])


# ---------------------- 核心修改：划分训练集和验证集 ----------------------

# 1. 初始化完整数据集（先过滤无效图片，再划分）
full_dataset = PlantImageDataset(
    csv_file=r"F:\计挑赛\train_labels.csv",
    image_dir=r"F:\计挑赛\train",
    transform=None  # 先不指定transform，划分后再分别赋值
)

# 2. 定义划分比例（验证集占10%，可调整为0.2即20%）
val_ratio = 0.1
val_size = int(val_ratio * len(full_dataset))
train_size = len(full_dataset) - val_size

# 3. 随机划分（设置seed保证每次划分结果一致）
torch.manual_seed(42)  # 固定随机种子，复现性强
train_subset, val_subset = random_split(
    dataset=full_dataset,
    lengths=[train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # 进一步确保划分稳定
)

# 4. 为子集指定专属预处理（关键：训练集增强，验证集不增强）
train_subset.dataset.transform = train_transform  # 训练子集用训练预处理
val_subset.dataset.transform = val_transform      # 验证子集用验证预处理

# 5. 创建训练集/验证集DataLoader
train_dataloader = DataLoader(
    train_subset,
    batch_size=32,
    shuffle=True,  # 训练集打乱
    pin_memory=True,
    drop_last=True  # 训练集丢弃不完整批次
)

val_dataloader = DataLoader(
    val_subset,
    batch_size=32,
    shuffle=False,  # 验证集不打乱（评估稳定）
    pin_memory=True,
    drop_last=False  # 验证集保留不完整批次（不浪费样本）
)

# 6. 打印划分结果
print(f"\n✅ 数据集划分完成：")
print(f"训练集样本数: {len(train_subset)} | 训练集批次数量: {len(train_dataloader)}")
print(f"验证集样本数: {len(val_subset)} | 验证集批次数量: {len(val_dataloader)}")

# 7. （可选）验证子集加载效果
def check_subset_loader(loader, name):
    print(f"\n🔍 验证{name}加载器：")
    for batch_idx, (images, labels) in enumerate(loader):
        print(f"批次 {batch_idx}: 图像形状 {images.shape}，标签形状 {labels.shape}")
        print(f"标签范围: {labels.min().item()} - {labels.max().item()}")
        if batch_idx >= 1:  # 仅验证前2个批次
            break

check_subset_loader(train_dataloader, "训练集")
check_subset_loader(val_dataloader, "验证集")

# 获取全局标签映射（划分后仍共用）
label_mapping = full_dataset.get_label_mapping()
print(f"\n总类别数量: {len(label_mapping)}")


