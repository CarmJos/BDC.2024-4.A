import os
import re

from PIL import Image
from torch.utils.data import Dataset

import config


class ImageClassifyDataset(Dataset):
    def __init__(self, paths: list, transform, id_range: list):
        self.path = paths
        self.transform = transform

        self.label_to_num = {}
        # with open("../map.txt", "r") as f:
        #     while line := f.readline():
        #         sec = line.split(" ")
        #         label = int(sec[0].strip())
        #         label_str = sec[1].strip()
        #         self.label_to_num[label_str] = label

        self.data = []
        label_pattern = re.compile(config.label_pattern)
        group_pattern = re.compile(config.group_pattern)
        id_pattern = re.compile(config.id_pattern)

        self.imgs = {}

        for directory in paths:
            for file in os.listdir(directory):
                img_path = directory + "/" + file

                matches = label_pattern.findall(file)
                if len(matches) < 2:
                    continue
                class_name = matches[0]
                subclass_name = matches[1]

                matches = group_pattern.findall(file)
                if not matches:
                    continue
                group_num = int(matches[0])

                matches = id_pattern.findall(file)
                if not matches:
                    continue
                id_num = int(matches[0][2])
                if id_num not in id_range:
                    continue

                self.data.append({
                    "label": group_num,
                    "url": img_path
                })

                self.imgs[group_num] = Image.open(img_path).convert("RGB")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_data = self.data[index]

        label = img_data["label"]
        img = self.imgs[label]

        img = self.transform(img)

        return img, label
