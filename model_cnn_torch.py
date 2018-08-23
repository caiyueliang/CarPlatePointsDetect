# encoding:utf-8
import os
import common
import random
import numpy as np
from PIL import Image
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T
from torchvision.transforms import functional
from torch.autograd import Variable
from torch.utils import data
import cv2


# 图片加载类
class MyDataset(data.Dataset):
    def __init__(self, root_dir, label_file, img_size, transforms=None, is_train=False):
        self.root_dir = root_dir
        records_txt = common.read_data(label_file, 'r')
        self.records = records_txt.split('\n')
        self.img_size = img_size
        self.is_train = is_train

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

        label = str_list[2:]
        label = map(float, label)
        label = np.array(label)

        if self.is_train:                                               # 训练模式，才做变换
            # img, label = self.RandomHorizontalFlip(img, label)        # 图片做随机水平翻转
            img, label = self.random_crop(img, label)
            # self.show_img(img, label)

        label = label * self.img_size / old_size
        if self.transforms:
            img = self.transforms(img)

        return img, label, img_file

    def __len__(self):
        return len(self.records)

    # 图片做随机水平翻转
    def RandomHorizontalFlip(self, img, label, p=0.5):
        if random.random() < p:
            w, h = img.size
            img = functional.hflip(img)
            for i in range(len(label)/2):
                label[2*i] = w - label[2*i]

        return img, label

    # 随机裁剪
    def random_crop(self, img, labels):
        # print('random_crop', labels)
        # mode = random.choice([None, 0.3, 0.5, 0.7, 0.9])
        # random.randrange(int(0.3*short_size), short_size)
        img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow('img_cv', img_cv)

        imh, imw, _ = img_cv.shape
        short_size = min(imw, imh)
        # print(imh, imw, short_size)

        left = min(labels[0], labels[4])
        top = min(labels[1], labels[3])
        right = max(labels[2], labels[6])
        bottom = max(labels[5], labels[7])
        # print('left, top, right, bottom', left, top, right, bottom)

        x1 = 0
        y1 = 0
        x2 = imw
        y2 = imh

        if random.random() < 0.5:
            rate = random.random()
            x1 = int(left * rate)
            labels[0] = labels[0] - int(left * rate)
            labels[2] = labels[2] - int(left * rate)
            labels[4] = labels[4] - int(left * rate)
            labels[6] = labels[6] - int(left * rate)

        if random.random() < 0.5:
            rate = random.random()
            y1 = int(top * rate)
            labels[1] = labels[1] - int(top * rate)
            labels[3] = labels[3] - int(top * rate)
            labels[5] = labels[5] - int(top * rate)
            labels[7] = labels[7] - int(top * rate)

        if random.random() < 0.5:
            rate = random.random()
            x2 = imw - int((imw - right) * rate)

        if random.random() < 0.5:
            rate = random.random()
            y2 = imh - int((imh - bottom) * rate)

        # print('x1, y1, x2, y2', x1, y1, x2, y2)
        img_cv = img_cv[y1:y2, x1:x2]
        # cv2.imshow('crop_img_cv', img_cv)
        # cv2.waitKey(0)

        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return img, labels

    def show_img(self, img, output):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        for i in range(len(output)/2):
            cv2.circle(img, (int(output[2*i]), int(output[2*i+1])), 3, (0, 0, 255), -1)

        cv2.imshow('show_img', img)
        cv2.waitKey(0)


# 自定义Loss
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def forward(self, outputs, targets):                            # mse：最小平方误差函数
        loss_list = []
        for output, target in zip(outputs, targets):
            print output
            print target
            loss = 0
            for x, y in zip(output, target):
                print x, y
                loss += torch.sqrt(x - y)
            loss_list.append(loss)
        return loss_list


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

    def load(self, name):
        print('[Load model] %s...' % name)
        self.load_state_dict(torch.load(name))

    def save(self, name):
        print('[Save model] %s ...' % name)
        torch.save(self.state_dict(), name)


class ModuleTrain:
    def __init__(self, train_path, test_path, model_file, model=CNN(), img_size=178, batch_size=8, lr=1e-3,
                 re_train=False, best_loss = 0.3):
        self.train_path = train_path
        self.test_path = test_path
        self.model_file = model_file
        self.img_size = img_size
        self.batch_size = batch_size
        self.re_train = re_train                        # 不加载训练模型，重新进行训练
        self.best_loss = best_loss                      # 最好的损失值，小于这个值，才会保存模型

        if torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False

        print('[ModuleCNN]')
        print('train_path: %s' % self.train_path)
        print('test_path: %s' % self.test_path)
        print('img_size: %d' % self.img_size)
        print('batch_size: %d' % self.batch_size)

        # 模型
        self.model = model

        if self.use_gpu:
            print('[use gpu] ...')
            self.model = self.model.cuda()

        # 加载模型
        if os.path.exists(self.model_file) and not self.re_train:
            self.load(self.model_file)

        # RandomHorizontalFlip
        self.transform_train = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        ])

        self.transform_test = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        # Dataset
        train_label = os.path.join(self.train_path, 'label.txt')
        train_dataset = MyDataset(self.train_path, train_label, self.img_size, self.transform_test, is_train=True)
        test_label = os.path.join(self.test_path, 'label.txt')
        test_dataset = MyDataset(self.test_path, test_label, self.img_size, self.transform_test, is_train=False)
        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        # self.loss = F.mse_loss
        self.loss = F.smooth_l1_loss

        self.lr = lr
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        pass

    def train(self, epoch, decay_epoch=40, save_best=True):
        print('[train] epoch: %d' % epoch)
        for epoch_i in range(epoch):

            if epoch_i >= decay_epoch and epoch_i % decay_epoch == 0:                   # 减小学习速率
                self.lr = self.lr * 0.1
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            print('================================================')
            for batch_idx, (data, target, _) in enumerate(self.train_loader):
                data, target = Variable(data), Variable(target)

                if self.use_gpu:
                    data = data.cuda()
                    target = target.cuda()

                # 梯度清0
                self.optimizer.zero_grad()

                # 计算损失
                output = self.model(data)
                loss = self.loss(output.type(torch.FloatTensor), target.type(torch.FloatTensor))

                # 反向传播计算梯度
                loss.backward()

                # 更新参数
                self.optimizer.step()

                # update
                if batch_idx == 0:
                    print('[Train] Epoch: {} [{}/{}]\tLoss: {:.6f}\tlr: {}'.format(epoch_i, batch_idx * len(data),
                        len(self.train_loader.dataset), loss.item()/self.batch_size, self.lr))

            test_loss = self.test()
            if save_best is True:
                if self.best_loss > test_loss:
                    self.best_loss = test_loss
                    str_list = self.model_file.split('.')
                    best_model_file = ""
                    for str_index in range(len(str_list)):
                        best_model_file = best_model_file + str_list[str_index]
                        if str_index == (len(str_list) - 2):
                            best_model_file += '_best'
                        if str_index != (len(str_list) - 1):
                            best_model_file += '.'
                    self.save(best_model_file)                                  # 保存最好的模型

        self.save(self.model_file)

    def test(self, show_img=False):
        test_loss = 0
        correct = 0

        # 测试集
        for data, target, img_files in self.test_loader:
            # print('[test] data.size: ', data.size())
            data, target = Variable(data), Variable(target)
            # print('[test] data.size: ', data.size())

            if self.use_gpu:
                data = data.cuda()
                target = target.cuda()

            output = self.model(data)
            # sum up batch loss
            if self.use_gpu:
                loss = self.loss(output, target.type(torch.cuda.FloatTensor))
            else:
                loss = self.loss(output, target.type(torch.FloatTensor))
            test_loss += loss.item()

            if show_img:
                for i in range(len(output[:, 1])):
                    self.show_img(img_files[i], output[i].cpu().detach().numpy(), target[i].cpu().detach().numpy())

        test_loss /= len(self.test_loader.dataset)
        print('[Test] set: Average loss: {:.4f}\n'.format(test_loss))
        return test_loss

    def load(self, name):
        print('[Load model] %s ...' % name)
        self.model.load_state_dict(torch.load(name))
        # self.model.load(name)

    def save(self, name):
        print('[Save model] %s ...' % name)
        torch.save(self.model.state_dict(), name)
        # self.model.save(name)

    def show_img(self, img_file, output, target):
        print(img_file)
        print(output)
        print(target)

        img = cv2.imread(img_file)
        h, w, c = img.shape
        for i in range(len(target)/2):
            cv2.circle(img, (int(target[2*i]*h/self.img_size), int(target[2*i+1]*h/self.img_size)), 3, (0, 255, 0), -1)

        for i in range(len(output)/2):
            cv2.circle(img, (int(output[2*i]*h/self.img_size), int(output[2*i+1]*h/self.img_size)), 3, (0, 0, 255), -1)

        cv2.imshow('show_img_1', img)
        cv2.waitKey(0)


# if __name__ == '__main__':
    # model = CNN()
    # data = Variable(torch.randn(1, 3, 178, 178))
    # x = model(data)
    # print('x', x.size())

    # train_dir = "/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_train"
    # test_dir = "/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_test"
    # model = ModuleCNN(train_dir, test_dir, "./1.h5")
    # model.train(1000)

    # img_dir = "/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_all"
    # label_file = "/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/label_all.txt"
    # my_dataset = MyDataset(img_dir, label_file, transform)
    # img, label = my_dataset[0]

    # im = Image.open('/home/caiyueliang/deeplearning/lpr-service/capture_service/capture_image/96.3_96.7_use_box/failed/481724_闽DK7103.jpg')
    # # 获得图像尺寸:
    # w, h = im.size
    # print(im.size)
    #
    # img = cv2.imread('/home/caiyueliang/deeplearning/lpr-service/capture_service/capture_image/96.3_96.7_use_box/failed/481724_闽DK7103.jpg')
    # print(img.shape)








