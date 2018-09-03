# coding=utf-8
import model_cnn_torch
import model_resnet_torch
from torchvision import models

if __name__ == '__main__':
    train_path = '../Data/car_finemap_detect/car_plate_train'
    test_path = '../Data/car_finemap_detect/car_plate_test'

    # FILE_PATH = './Model/model_params.pkl'
    # model_train = model_cnn_torch.ModuleTrain(train_path, test_path, FILE_PATH, img_size=178, lr=1e-4)

    FILE_PATH = './Model/resnet18_params.pkl'
    model = models.resnet18(num_classes=8)
    model_train = model_cnn_torch.ModuleTrain(train_path, test_path, FILE_PATH, model=model, batch_size=16, img_size=224, lr=1e-3)

    # FILE_PATH = './Model/model_resnet34_params.pkl'
    # model = models.resnet34(num_classes=8)
    # model_train = model_cnn_torch.ModuleTrain(train_path, test_path, FILE_PATH, model=model, batch_size=16, img_size=224, lr=1e-3)

    # FILE_PATH = './Model/model_resnet50_params.pkl'
    # model = models.resnet50(num_classes=8)
    # model_train = model_cnn_torch.ModuleTrain(train_path, test_path, FILE_PATH, model=model, batch_size=16, img_size=224, lr=1e-3)

    model_train.train(300, 120)
    model_train.test(show_img=True)
