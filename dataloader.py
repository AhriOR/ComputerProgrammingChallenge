import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

image_category_dict={}
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

        # 打印数据集信息
        print(f"数据集大小: {len(self.data_frame)}")
        print(f"类别数量: {self.data_frame['category_id'].nunique()}")
        print(f"类别分布:\n{self.data_frame['category_id'].value_counts().head()}")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 获取数据
        filename = self.data_frame.iloc[idx]['filename']
        category_id = self.data_frame.iloc[idx]['category_id']
        chinese_name = self.data_frame.iloc[idx]['chinese_name']
        english_name = self.data_frame.iloc[idx]['english_name']

        for id,name in zip(category_id,chinese_name):
            image_category_dict[str(id)]=name

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

        # 返回图像和类别ID
        return image, category_id


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
dataset = PlantImageDataset(
    csv_file=r"F:\计挑赛\train_labels.csv",
    image_dir=r"F:\计挑赛\train",
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)