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
import time


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

        # print('img_file', img_file)
        img = Image.open(img_file)
        # img = cv2.imread(img_file)

        label = str_list[2:]
        label = map(float, label)
        label = np.array(label)

        if self.is_train:                                               # 训练模式，才做变换
            # img, label = self.RandomHorizontalFlip(img, label)        # 图片做随机水平翻转
            img, label = self.random_crop(img, label)                   # 图片做随机裁剪
            # self.show_img(img, label)

        old_size = img.size[0]
        # old_size = img.shape[0]
        label = label * self.img_size / old_size

        # img = cv2.resize(img, (self.img_size, self.img_size))
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

        # short_size = min(imw, imh)
        # print(imh, imw, short_size)

        left = min(labels[0], labels[4])
        top = min(labels[1], labels[3])
        min_left = min(left, top)

        right = min(imw-labels[2], imw-labels[6])
        bottom = min(imh-labels[5], imh-labels[7])
        min_right = min(right, bottom)

        # print('left, top, right, bottom', left, top, right, bottom)
        # print('min_left, min_right', min_left, min_right)

        x1 = 0
        y1 = 0
        x2 = imw
        y2 = imh

        if random.random() < 0.5:
            rate = random.random()
            crop = int(min_left * rate)
            x1 = crop
            labels[0] = labels[0] - crop
            labels[2] = labels[2] - crop
            labels[4] = labels[4] - crop
            labels[6] = labels[6] - crop

            y1 = crop
            labels[1] = labels[1] - crop
            labels[3] = labels[3] - crop
            labels[5] = labels[5] - crop
            labels[7] = labels[7] - crop

        if random.random() < 0.5:
            rate = random.random()
            crop = int(min_right * rate)
            x2 = imw - crop
            y2 = imh - crop

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


class ModuleTrain:
    def __init__(self, train_path, test_path, model_file, model, img_size=224, batch_size=16, lr=1e-3,
                 re_train=False, best_loss=10, use_gpu=False):
        self.train_path = train_path
        self.test_path = test_path
        self.model_file = model_file
        self.img_size = img_size
        self.batch_size = batch_size
        self.re_train = re_train                        # 不加载训练模型，重新进行训练
        self.best_loss = best_loss                      # 最好的损失值，小于这个值，才会保存模型
        self.use_gpu = False

        if use_gpu is True:
            print("gpu available: %s" % str(torch.cuda.is_available()))
            if torch.cuda.is_available():
                self.use_gpu = True
            else:
                self.use_gpu = False

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
            train_loss = 0.0
            if epoch_i >= decay_epoch and epoch_i % decay_epoch == 0:                   # 减小学习速率
                self.lr = self.lr * 0.1
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            print('================================================')
            self.model.train()
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
                train_loss += loss.item()
                # 反向传播计算梯度
                loss.backward()
                # 更新参数
                self.optimizer.step()

            train_loss /= len(self.train_loader)
            print('[Train] Epoch: {} \tLoss: {:.6f}\tlr: {}'.format(epoch_i, train_loss, self.lr))

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
        test_loss = 0.0
        correct = 0

        time_start = time.time()
        # 测试集
        self.model.eval()
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
        time_end = time.time()
        test_loss /= len(self.test_loader)
        time_avg = float(time_end - time_start) / float(len(self.test_loader.dataset))

        print('[Test] set: Average loss: {:.6f} time: {:.6f}\n'.format(test_loss, time_avg))
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
        # print(img_file)
        # print(output)
        # print(target)

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








