import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from net import VGG16
from src import config


class Trainer:
    def __init__(self, train_set, test_set, train_times, learning_rate):
        self.train_dataset = train_set
        self.test_dataset = test_set
        self.train_times = train_times
        self.learning_rate = learning_rate
        self.model = None

    def start(self):
        train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        model = VGG16(28)
        self.model = model
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            model = model.cuda()
            print("using cuda")
        else:
            print("not using cuda")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)

        # 开始训练
        for epoch in range(self.train_times):
            print('-' * 32)
            print(f'epoch {epoch + 1}')
            running_loss = 0.0
            running_acc = 0.0
            for i, data in enumerate(train_loader, 1):
                img, label = data

                with torch.no_grad():
                    if cuda_available:
                        img = img.cuda()
                        label = label.cuda()
                    else:
                        img = Variable(img)
                        label = Variable(label)

                # 向前传播
                out = model(img)
                loss = criterion(out, label)
                running_loss += loss.item() * label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == label).sum()
                running_acc += num_correct.item()
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Finish {epoch + 1}/{self.train_times} epoch, '
                  f'Loss: {running_loss / (len(self.train_dataset)) :.6f} Acc:{running_acc / (len(self.train_dataset)) * 100 :.6f}%')

            # 测试
            model.eval()
            eval_loss = 0
            eval_acc = 0
            for data in val_loader:
                img, label = data
                with torch.no_grad():
                    if cuda_available:
                        img = Variable(img).cuda()
                        label = Variable(label).cuda()
                    else:
                        img = Variable(img)
                        label = Variable(label)
                out = model(img)
                loss = criterion(out, label)
                eval_loss += loss.item() * label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == label).sum()
                eval_acc += num_correct.item()
            print(f'Test Loss: {eval_loss / (len(self.test_dataset)) :.6f}, Acc: {eval_acc / (len(self.test_dataset)) * 100 :.6f}%')


            save_epoch = [20, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
            if epoch in save_epoch:
                self.save(f'../model{epoch}.pth')

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
