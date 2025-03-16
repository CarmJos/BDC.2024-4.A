import os
import re

from PIL import Image
from torch.utils.data import Dataset

from src.labeler import Labeler


class ImageClassifyDataset(Dataset):
    def __init__(self, paths: list, transform, labeler: Labeler, id_range: list):
        self.path = paths
        self.transform = transform

        self.data = []

        self.img_cache = {}

        for directory in paths:
            for file in os.listdir(directory):
                img_path = directory + "/" + file

                group_num = labeler.label(file, id_range)
                if group_num < 0:
                    continue

                self.data.append({
                    "label": group_num,
                    "url": img_path
                })

                # self.img_cache[img_path] = Image.open(img_path).convert("RGB")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_data = self.data[index]
        img_path = img_data["url"]

        label = img_data["label"]
        if img_path in self.img_cache:
            img = self.img_cache[img_path]
        else:
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
