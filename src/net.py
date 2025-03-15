import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, n):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # Block 1 (64 filters, 3×3 filters, same padding)
            # conv 1-1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # conv 1-2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # max pool (2×2, stride 2)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2 (128 filters, 3×3 filters, same padding)
            # conv 2-1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # conv 2-2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # max pool (2×2, stride 2)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3 (256 filters, 3×3 filters, same padding)
            # conv 3-1
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv 3-2
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # conv 3-3
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # max pool (2×2, stride 2)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4 (512 filters, 3×3 filters, same padding)
            # conv 4-1
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv 4-2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv 4-3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # max pool (2×2, stride 2)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5 (512 filters, 3×3 filters, same padding)
            # conv 5-1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv 5-2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # conv 5-3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # max pool (2×2, stride 2)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layer
        self.classifier = nn.Sequential(
            # Flattening
            nn.Linear(512 * 7 * 7, 4096),
            # nn.BatchNorm1d(4096),  # 1D BN
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            # nn.BatchNorm1d(4096),  # 1D BN
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, n)
        )


def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)  # 展开特征图
    x = self.classifier(x)
    return x
