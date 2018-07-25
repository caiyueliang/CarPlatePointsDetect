# encoding:utf-8
import os
import common
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T
from torchvision import datasets
from torch.autograd import Variable
from torch.utils import data

img_size = 178

transform = T.Compose([
    T.Resize(img_size),
    T.ToTensor(),
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


# 图片加载类
class MyDataset(data.Dataset):
    def __init__(self, root_dir, label_file, transforms=None):
        self.root_dir = root_dir
        records_txt = common.read_data(label_file, 'r')
        self.records = records_txt.split('\n')

        # imgs = os.listdir(root)
        # self.imgs = [os.path.join(root, img) for img in imgs]
        # self.label_path = label_path
        self.transforms = transforms

    def __getitem__(self, index):
        record = self.records[index]
        str_list = record.split(" ")
        img_file = os.path.join(self.root_dir, str_list[0])

        img = Image.open(img_file)
        old_size = img.size[0]
        if self.transforms:
            img = self.transforms(img)

        label = str_list[2:]
        label = map(float, label)
        label = np.array(label)
        label = label * img_size / old_size
        # label = label / old_size
        print(label)
        # label = label.reshape(1, 8)
        # print(label)

        return img, label

    def __len__(self):
        return len(self.records)


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
        # print('conv1', x1.size())
        x1 = F.relu(x1)
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2, padding=0)
        # print('max_pool2d', x1.size())

        x2 = self.batch_2(x1)
        x2 = self.conv_2(x2)
        # print('conv2', x2.size())
        x2 = F.relu(x2)
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2, padding=0)
        # print('max_pool2d', x2.size())

        x3 = self.batch_3(x2)
        x3 = self.conv_3(x3)
        # print('conv3', x3.size())
        x3 = F.relu(x3)
        x3 = F.max_pool2d(x3, kernel_size=2, stride=2, padding=0)
        # print('max_pool2d', x3.size())

        x4 = self.dropout_1(x3)
        # x4 = x4.view(x4.size(0), -1)
        x4 = x4.view(-1, 64*20*20)
        # print('view', x4.size())
        x4 = F.relu(self.fc1(x4))
        # print('fc1', x4.size())
        x4 = F.relu(self.fc2(x4))
        # print('fc2', x4.size())

        output = x4
        return output


class ModuleCNN():
    def __init__(self, train_path, test_path, model_file, img_size=178, batch_size=8, epoch_num=30):
        self.model = CNN()

        self.train_path = train_path
        self.test_path = test_path
        self.model_file = model_file
        # self.train_samples = len(common.get_img_files(train_path))
        # self.test_samples = len(common.get_img_files(test_path))
        self.img_size = img_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num

        print('[ModuleCNN]')
        print('train_path: %s' % self.train_path)
        print('test_path: %s' % self.test_path)
        print('img_size: %d' % self.img_size)
        print('batch_size: %d' % self.batch_size)
        print('epoch_num: %d' % self.epoch_num)

        # MNIST Dataset
        # train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
        # test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor())
        train_dir = "/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_train"
        train_label = "/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/label_train.txt"
        train_dataset = MyDataset(train_dir, train_label, transform)

        test_dir = "/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_test"
        test_label = "/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/label_test.txt"
        test_dataset = MyDataset(test_dir, test_label, transform)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        pass

    def train(self, epoch):
        # 每次输入barch_idx个数据
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = Variable(data), Variable(target)

            optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
            optimizer.zero_grad()
            output = self.model(data)
            print output
            print target
            # loss
            loss = F.nll_loss(output, target)
            loss.backward()
            # update
            optimizer.step()
            if batch_idx % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.data[0]))

    def test(self):
        test_loss = 0
        correct = 0

        # 测试集
        for data, target in self.test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target).data[0]
            # get the index of the max
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))


if __name__ == '__main__':
    # model = CNN()
    # data = Variable(torch.randn(10, 3, 178, 178))
    # x = model(data)
    # print('x', x.size())

    model = ModuleCNN("./", "./", "./1.h5")
    model.train(10)

    # img_dir = "/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_all"
    # label_file = "/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/label_all.txt"
    # my_dataset = MyDataset(img_dir, label_file, transform)
    # img, label = my_dataset[0]









