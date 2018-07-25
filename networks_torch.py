# encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
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


class ModuleCNN(nn.Module):
    def __init__(self, train_path, test_path, model_file, img_size=178, batch_size=5, epoch_num=30):
        self.model = CNN()

        self.train_path = train_path
        self.test_path = test_path
        self.model_file = model_file
        # self.train_samples = len(common.get_img_files(train_path))
        # self.test_samples = len(common.get_img_files(test_path))
        self.img_size = img_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num

        print('[ModelCNN]')
        print('train_samples: %d' % self.train_samples)
        print('test_samples: %d' % self.test_samples)
        print('img_size: %d' % self.img_size)
        print('batch_size: %d' % self.batch_size)
        print('epoch_num: %d' % self.epoch_num)

        # MNIST Dataset
        train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor())

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        pass

    def train(self, epoch):
        # 每次输入barch_idx个数据
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = Variable(data), Variable(target)

            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
            optimizer.zero_grad()
            output = model(data)
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
            output = model(data)
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
    model = CNN()
    # print model
    data = Variable(torch.randn(10, 3, 178, 178))
    x = model(data)
    print('x', x.size())

