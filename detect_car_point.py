# coding=utf-8
from argparse import ArgumentParser
from model_train import ModuleTrain
from models import model_resnet_torch
from models import model_resnet_squeeze
from torchvision import models


def parse_argvs():
    parser = ArgumentParser(description='car_classifier')
    parser.add_argument('--train_path', type=str, help='train dataset path', default='../Data/car_finemap_detect_new/car_plate_train')
    parser.add_argument('--test_path', type=str, help='test dataset path', default='../Data/car_finemap_detect_new/car_plate_test')

    parser.add_argument("--output_model_path", type=str, help="output model path", default='./models/resnet18_params_sq.pkl')
    parser.add_argument('--classes_num', type=int, help='classes num', default=8)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--img_size', type=int, help='imgsize', default=224)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)

    input_args = parser.parse_args()
    print(input_args)
    return input_args


if __name__ == '__main__':
    args = parse_argvs()

    train_path = args.train_path
    test_path = args.test_path

    # FILE_PATH = './Model/model_params.pkl'
    # model_train = model_cnn_torch.ModuleTrain(train_path, test_path, FILE_PATH, img_size=178, lr=1e-4)

    # FILE_PATH = './Model/my_resnet18_params.pkl'
    # model = model_resnet_torch.ResNet18(num_classes=8)
    # model_train = model_cnn_torch.ModuleTrain(train_path, test_path, FILE_PATH, model=model, batch_size=16, img_size=224, lr=1e-2)

    # FILE_PATH = './Model/resnet18_params.pkl'
    # model = models.resnet18(num_classes=8)
    # model_train = model_cnn_torch.ModuleTrain(train_path, test_path, FILE_PATH, model=model, batch_size=32, img_size=224, lr=1e-3)

    # FILE_PATH = './Model/Squeezenet1_1.pkl'
    # model = models.squeezenet1_1(num_classes=8)
    # model_train = model_cnn_torch.ModuleTrain(train_path, test_path, FILE_PATH, model=model, batch_size=200, img_size=224, lr=1e-2)

    output_model_path = args.output_model_path
    num_classes = args.classes_num
    batch_size = args.batch_size
    img_size = args.img_size
    lr = args.lr

    print('train_path: %s' % train_path)
    print('test_path: %s' % test_path)
    print('output_model_path: %s' % output_model_path)
    print('num_classes: %d' % num_classes)
    print('img_size: %d' % img_size)
    print('batch_size: %d' % batch_size)
    print('lr: %s' % lr)

    model = model_resnet_squeeze.resnet18(num_classes=num_classes)
    model_train = ModuleTrain(train_path=train_path, test_path=test_path, model_file=output_model_path, model=model,
                              batch_size=batch_size, img_size=img_size, lr=lr)

    model_train.train(200, 80)
    # model_train.test(show_img=True)
