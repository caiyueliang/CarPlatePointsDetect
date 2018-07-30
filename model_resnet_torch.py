# coding=utf-8
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet18, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 重复的layer，分别有3,4,6,3个residual block
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        self.layer4 = self._make_layer(512, 512, 2, stride=2)

        # 分类用的全连接（这边不是分类）
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channel, out_channel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        layers = []
        layers.append(ResidualBlock(in_channel, out_channel, stride, shortcut))

        for ii in range(1, block_num):
            layers.append(ResidualBlock(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load(self, name):
        print('[Load model] %s...' % name)
        self.load_state_dict(torch.load(name))

    def save(self, name):
        print('[Save model] %s ...' % name)
        torch.save(self.state_dict(), name)

from torchvision import models

if __name__ == '__main__':
    model = ResNet18()
    print model
    input = Variable(torch.randn(4, 3, 224, 224))
    print (input.size())
    o = model(input)
    print (o.size())

    model_1 = models.resnet34(num_classes=8)
    print model_1
    o = model_1(input)
    print (o.size())
