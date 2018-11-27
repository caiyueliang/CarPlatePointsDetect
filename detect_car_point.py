# coding=utf-8
import os
from argparse import ArgumentParser
from model_train import ModuleTrain
from models import model_resnet_torch
from models import model_resnet_squeeze
from models import model_mobilenet_v2
from torchvision import models


def parse_argvs():
    parser = ArgumentParser(description='car_classifier')
    parser.add_argument('--train_path', type=str, help='train dataset path', default='../Data/car_finemap_detect_new/car_plate_train')
    parser.add_argument('--test_path', type=str, help='test dataset path', default='../Data/car_finemap_detect_new/car_plate_test')

    parser.add_argument('--model_name', type=str, help='model name', default='mobilenet_v2')
    # parser.add_argument('--model_name', type=str, help='model name', default='resnet18_sq')
    parser.add_argument("--output_model_path", type=str, help="output model path", default='./checkpoints')
    parser.add_argument('--classes_num', type=int, help='classes num', default=8)
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--img_size', type=int, help='img size', default=224)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)

    # mobilenet 使用参数
    parser.add_argument('--downsampling', type=int, help='down sampling: 8 16 32', default=8)
    parser.add_argument('--num_channels', type=int, help='num channels', default=3)
    parser.add_argument('--kernel_size', type=int, help='kernel size', default=3)
    parser.add_argument('--width_multiplier', type=float, help='width multiplier', default=0.125)
    parser.add_argument('--dropout_prob', type=float, help='dropout prob', default=0.6)
    parser.add_argument('--img_height', type=int, help='img_height', default=224)
    parser.add_argument('--img_width', type=int, help='img_width', default=224)
    parser.add_argument('--cuda', type=bool, help='use gpu', default=True)

    input_args = parser.parse_args()
    print(input_args)
    return input_args


if __name__ == '__main__':
    args = parse_argvs()

    train_path = args.train_path
    test_path = args.test_path

    if 'mobilenet_v2' in args.model_name:
        model = model_mobilenet_v2.MobileNetV2(args)
        output_model_path = os.path.join(args.output_model_path, 'mobilenet_v2_params.pkl')
    elif 'resnet18_sq' in args.model_name:
        model = model_resnet_squeeze.resnet18(num_classes=args.classes_num)
        output_model_path = os.path.join(args.output_model_path, 'resnet18_sq_params.pkl')
    elif 'inception_v3' in args.model_name:
        model = models.Inception3(num_classes=args.classes_num)
        output_model_path = os.path.join(args.output_model_path, 'inception_v3_params.pkl')
    else:
        model = model_resnet_torch.resnet18(num_classes=args.classes_num)
        output_model_path = os.path.join(args.output_model_path, 'resnet18_params.pkl')

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

    model_train = ModuleTrain(train_path=train_path, test_path=test_path, model_file=output_model_path, model=model,
                              batch_size=batch_size, img_size=img_size, lr=lr)

    model_train.train(200, 80)
    # model_train.test(show_img=True)
