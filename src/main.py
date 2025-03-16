import os

import torchvision

from src import dataset, config
from src.labeler import SedimentaryLabeler, StoneLabeler
from src.trainer import Trainer


# 普通变换
val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),  # 缩放图像到 224x224
    torchvision.transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
# 数据增强变换
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),  # 缩放图像到 224x224
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平镜像
    torchvision.transforms.RandomGrayscale(),
    torchvision.transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])


def train_sedimentary():
    print("reading files...")
    # 数据集
    data_path = ["../data/source/南京大学教学沉积岩薄片照片数据集/"]

    labeler = SedimentaryLabeler()

    train_dataset = dataset.ImageClassifyDataset(data_path, train_transform, labeler, [2, 3, 4, 5, 6, 9])
    test_dataset = dataset.ImageClassifyDataset(data_path, val_transform, labeler, [7, 8])

    print(f"size of train sets: {len(train_dataset)}")
    print(f"size of test sets: {len(test_dataset)}")

    # 开始训练
    trainer = Trainer(train_dataset, test_dataset, 11, config.num_epochs, config.learning_rate)
    trainer.start()
    # 保存模型
    trainer.save('../model_sedimentary.pth')


def train_stone_classification():
    print("reading files...")
    # 数据集
    data_path = ["../data/source/南京大学变质岩教学薄片照片数据集/", "../data/source/南京大学火成岩教学薄片照片数据集/", "../data/source/南京大学教学沉积岩薄片照片数据集/"]

    labeler = StoneLabeler()

    train_dataset = dataset.ImageClassifyDataset(data_path, train_transform, labeler, [2, 3, 4, 5, 6, 9])
    test_dataset = dataset.ImageClassifyDataset(data_path, val_transform, labeler, [7, 8])

    print(f"size of train sets: {len(train_dataset)}")
    print(f"size of test sets: {len(test_dataset)}")

    # 开始训练
    trainer = Trainer(train_dataset, test_dataset, 3, config.num_epochs, config.learning_rate)
    trainer.start()
    # 保存模型
    trainer.save('../model_stone.pth')


def solution3():


def main():
    # train_sedimentary()
    train_stone_classification()


if __name__ == '__main__':
    main()
