# encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable


# model.add(Conv2D(32, (3, 3), input_shape=(self.img_size, self.img_size, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(BatchNormalization())
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(8))
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.batch_1 = nn.BatchNorm2d(3)
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.batch_2 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.batch_3 = nn.BatchNorm2d(32)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.dropout_1 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=64*20*20, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=8)

    def forward(self, x):
        x1 = self.batch_1(x)
        x1 = self.conv_1(x1)
        print('conv1', x1.size())
        x1 = F.relu(x1)
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2, padding=0)
        print('max_pool2d', x1.size())

        x2 = self.batch_2(x1)
        x2 = self.conv_2(x2)
        print('conv2', x2.size())
        x2 = F.relu(x2)
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2, padding=0)
        print('max_pool2d', x2.size())

        x3 = self.batch_3(x2)
        x3 = self.conv_3(x3)
        print('conv3', x3.size())
        x3 = F.relu(x3)
        x3 = F.max_pool2d(x3, kernel_size=2, stride=2, padding=0)
        print('max_pool2d', x3.size())

        x4 = self.dropout_1(x3)
        x4 = x4.view(x4.size(0), -1)
        print('view', x4.size())
        x4 = F.relu(self.fc1(x4))
        print('fc1', x4.size())
        x4 = F.relu(self.fc2(x4))
        print('fc2', x4.size())

        output = x4
        return output


# class FaceBox(nn.Module):
#     input_size = 1024
#
#     def __init__(self):
#         super(FaceBox, self).__init__()
#
#         #model
#         self.conv1 = nn.Conv2d(3, 24, kernel_size=7, stride=4, padding=3)
#         self.conv2 = nn.Conv2d(48, 64, kernel_size=5, stride=2, padding=2)
#
#         self.inception1 = Inception()
#         self.inception2 = Inception()
#         self.inception3 = Inception()
#
#         self.conv3_1 = nn.Conv2d(128, 128, kernel_size=1)
#         self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
#         self.conv4_1 = nn.Conv2d(256, 128, kernel_size=1)
#         self.conv4_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
#
#         self.multilbox = MultiBoxLayer()
#
#     def forward(self, x):
#         hs = []
#
#         x = self.conv1(x)
#         print('conv1', x.size())
#         x = torch.cat((F.relu(x), F.relu(-x)), 1)
#         print('CReLu', x.size())
#
#         x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
#         print('max_pool2d', x.size())
#         x = self.conv2(x)
#         print('conv2', x.size())
#         x = torch.cat((F.relu(x), F.relu(-x)), 1)
#         print('CReLu', x.size())
#         x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
#         print('max_pool2d', x.size())
#
#         x = self.inception1(x)
#         print('inception1', x.size())
#         x = self.inception2(x)
#         print('inception2', x.size())
#         x = self.inception3(x)
#         print('inception3', x.size())
#         hs.append(x)
#
#         x = self.conv3_1(x)
#         print('conv3_1', x.size())
#         x = self.conv3_2(x)
#         print('conv3_2', x.size())
#         hs.append(x)
#
#         x = self.conv4_1(x)
#         print('conv4_1', x.size())
#         x = self.conv4_2(x)
#         print('conv4_2', x.size())
#         hs.append(x)
#         loc_preds, conf_preds = self.multilbox(hs)
#
#         return loc_preds, conf_preds


if __name__ == '__main__':
    model = CNN()
    # print model
    data = Variable(torch.randn(1, 3, 178, 178))
    x = model(data)
    print('x', x.size())

