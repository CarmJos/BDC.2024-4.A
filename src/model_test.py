import os

import numpy as np
import torchvision
import torch
from PIL import Image

from src.labeler import SedimentaryLabeler
from torch.utils.data import DataLoader

from src.net import VGG16


def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),  # 缩放图像到 224x224
        torchvision.transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    model = VGG16(n=11)
    model.load_state_dict(torch.load("../model_sedimentary.pth"))
    model.eval()

    with torch.no_grad():
        for f in os.listdir("../data/test/"):
            image = Image.open("../data/test/" + f).convert("RGB")
            image = transform(image).cuda()
            image = image.unsqueeze(0)
            output = model(image)
            _, predicted = torch.max(output, 1)
            print(f"{f}={predicted}")


if __name__ == '__main__':
    main()
