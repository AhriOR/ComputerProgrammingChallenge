import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

image_category_dict = {}


class PlantImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file: CSV文件路径
            image_dir: 图像目录路径
            transform: 数据预处理
        """
        self.image_dir = image_dir
        self.transform = transform

        # 读取CSV文件
        try:
            self.data_frame = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.data_frame = pd.read_csv(csv_file, encoding='gbk')
            except:
                self.data_frame = pd.read_csv(csv_file, encoding='latin-1')

        # 创建标签编码器并转换标签
        self.label_encoder = LabelEncoder()
        self.data_frame['encoded_label'] = self.label_encoder.fit_transform(self.data_frame['category_id'])

        # 打印数据集信息
        print(f"数据集大小: {len(self.data_frame)}")
        print(f"类别数量: {self.data_frame['encoded_label'].nunique()}")
        print(f"标签范围: {self.data_frame['encoded_label'].min()} - {self.data_frame['encoded_label'].max()}")
        print(f"类别分布:\n{self.data_frame['encoded_label'].value_counts().head()}")

        # 打印一些原始类别和编码后的对应关系
        print("\n类别映射示例:")
        sample_mapping = self.data_frame[['category_id', 'encoded_label']].drop_duplicates().head(10)
        for _, row in sample_mapping.iterrows():
            print(f"  原始ID: {row['category_id']} -> 编码后: {row['encoded_label']}")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 获取数据
        filename = self.data_frame.iloc[idx]['filename']
        category_id = self.data_frame.iloc[idx]['category_id']
        encoded_label = self.data_frame.iloc[idx]['encoded_label']
        chinese_name = self.data_frame.iloc[idx]['chinese_name']
        english_name = self.data_frame.iloc[idx]['english_name']

        # 存储类别映射信息
        image_category_dict[str(category_id)] = chinese_name

        # 构建图像路径
        img_path = os.path.join(self.image_dir, filename)

        # 加载图像
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {e}")
            # 创建黑色图像作为替代
            image = Image.new('RGB', (224, 224), color='black')

        # 数据预处理
        if self.transform:
            image = self.transform(image)

        # 返回图像和编码后的类别标签
        return image, encoded_label

    def get_label_mapping(self):
        """获取标签映射关系"""
        return dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 创建数据集
dataset = PlantImageDataset(
    csv_file=r"F:\计挑赛\train_labels.csv",
    image_dir=r"F:\计挑赛\train",
    transform=transform
)

# 获取标签映射（用于后续预测时解码）
label_mapping = dataset.get_label_mapping()
print(f"\n总类别数量: {len(label_mapping)}")

# 创建数据加载器
train_dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True
)


# 验证数据加载
def check_dataloader(dataloader):
    """检查数据加载器中的标签范围"""
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"批次 {batch_idx}: 图像形状 {images.shape}, 标签形状 {labels.shape}")
        print(f"标签范围: {labels.min().item()} - {labels.max().item()}")

        # 确保标签在有效范围内
        assert labels.min() >= 0, f"发现负标签: {labels.min()}"
        assert labels.max() < 100, f"发现超出范围的标签: {labels.max()}"

        if batch_idx >= 2:  # 只检查前几个批次
            break


print("\n检查数据加载器:")
check_dataloader(train_dataloader)