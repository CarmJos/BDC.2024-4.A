import os
import re

import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset

import config


class ImageClassifyDataset(Dataset):
    def __init__(self, paths: list, augment: bool, group_range: list):
        self.path = paths
        self.augment = augment

        # 普通变换
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # 缩放图像到 224x224
            torchvision.transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
        # 数据增强变换
        self.augmented_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # 缩放图像到 224x224
            torchvision.transforms.RandomHorizontalFlip(p=0.5),  # 随机水平镜像，概率为 50%
            torchvision.transforms.RandomVerticalFlip(p=0.5),  # 随机竖直镜像，概率为 50%
            torchvision.transforms.RandomRotation(degrees=30),  # 随机旋转，角度范围为 -30 到 30 度
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机调整亮度、对比度和饱和度
            torchvision.transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

        self.data = []
        label_pattern = re.compile(config.label_pattern)
        id_pattern = re.compile(config.id_pattern)

        for directory in paths:
            for file in os.listdir(directory):
                img_path = directory + "/" + file

                matches = label_pattern.findall(file)
                if len(matches) < 2:
                    continue
                class_name = matches[0]
                subclass_name = matches[1]

                matches = id_pattern.findall(file)
                if not matches:
                    continue
                group_num = int(matches[0][0])
                if group_num not in group_range:
                    continue

                augment_loop = 1 if not augment else config.num_augmented_samples + 1
                for augment_index in range(augment_loop):
                    self.data.append({
                        "class": class_name,
                        "label": subclass_name,
                        "url": img_path,
                        "augment_index": augment_index
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_data = self.data[index]
        img_path = img_data["url"]
        img = Image.open(img_path).convert("RGB")
        augment_index = img_data["augment_index"]

        label = img_data["label"]

        if not self.augment or augment_index == 0:
            img = self.transform(img)
        else:
            img = self.augmented_transform(img)

        return img, label
