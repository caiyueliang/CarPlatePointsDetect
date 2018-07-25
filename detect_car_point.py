# coding=utf-8
import model_cnn
import networks_torch




if __name__ == '__main__':
    FILE_PATH = './CNN_model_final.h5'
    train_path = '/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_train'
    test_path = '/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_test'

    models = model_cnn.ModelCNN(train_path, test_path, FILE_PATH)

    my_model = models.get_model()
    # models.load(my_model)
    models.train(my_model)
    models.save(my_model)
