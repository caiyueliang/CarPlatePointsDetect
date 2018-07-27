# coding=utf-8
import model_cnn_torch
import model_resnet_torch

if __name__ == '__main__':
    train_path = '/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_train'
    test_path = '/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_test'

    # FILE_PATH = './Model/model_params.pkl'
    # model_train = model_cnn_torch.ModuleTrain(train_path, test_path, FILE_PATH, img_size=178, lr=1e-4)

    FILE_PATH = './Model/model_resnet34_params.pkl'
    model = model_resnet_torch.ResNet(num_classes=8)
    model_train = model_cnn_torch.ModuleTrain(train_path, test_path, FILE_PATH, model=model, batch_size=1, img_size=224, lr=1e-4)

    model_train.train(100)
    model_train.test(show_img=True)
