# coding=utf-8
import model_cnn_torch


if __name__ == '__main__':
    FILE_PATH = './Model/model_params.pkl'
    train_path = '/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_train'
    test_path = '/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_test'

    model = model_cnn_torch.ModuleCNN(train_path, test_path, FILE_PATH, use_gpu=True)
    model.train(200)
