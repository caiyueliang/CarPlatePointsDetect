# coding=utf-8
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator,img_to_array
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D,ZeroPadding2D
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import SGD
import numpy as np
import cv2
from keras.callbacks import *
import keras
import common as common


class ModelCNN(object):
    def __init__(self, train_path, test_path, model_file, img_size=178, batch_size=8, epoch_num=50):
        self.train_path = train_path
        self.test_path = test_path
        self.model_file = model_file
        self.train_samples = len(common.get_files(train_path))
        self.test_samples = len(common.get_files(test_path))

        self.img_size = img_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        pass

    def get_model(self):
        model = Sequential()            # 178*178*3
        model.add(Conv2D(32, (3, 3), input_shape=(self.img_size, self.img_size, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8))
        model.summary()
        return model

    def train(self, model):
        # print(lable.shape)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # optimizer = SGD(lr=0.03, momentum=0.9, nesterov=True)
        # model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # epoch_num = 10

        learning_rate = np.linspace(0.03, 0.01, self.epoch_num)
        change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
        check_point = ModelCheckpoint('CNN_model_final.h5', monitor='val_loss', verbose=0, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=1)

        model.fit_generator(self.data_label(self.train_path), callbacks=[check_point, early_stop, change_lr],
                            samples_per_epoch=int(self.train_samples // self.batch_size), epochs=self.epoch_num,
                            validation_steps=int(self.test_samples // self.batch_size),
                            validation_data=self.data_label(self.test_path))

        # model.fit(traindata, trainlabel, batch_size=32, epochs=50,
        #           validation_data=(testdata, testlabel))
        model.evaluate_generator(self.data_label(self.test_samples))

    def data_label(self, path):
        f = open(os.path.join(path, "label.txt"), "r")
        j = 0
        i = -1
        # datalist = []
        # labellist = []

        while True:
            for line in f.readlines():
                i += 1
                j += 1
                a = line.replace("\n", "")
                b = a.split(" ")
                label = b[2:]
                print(label)

                # 对标签进行归一化（不归一化也行）
                # for num in b[1:]:
                #     lab = int(num) / 255.0
                #     labellist.append(lab)
                # lab = labellist[i * 10:j * 10]
                img_name = os.path.join(path, b[0])
                img = cv2.imread(img_name)
                img = cv2.resize(img, (self.img_size, self.img_size))
                print(img.shape)
                images = img_to_array(img).astype('float32')

                # images = load_img(img_name)
                # images = img_to_array(images).astype('float32')

                # 对图片进行归一化（不归一化也行）
                # images /= 255.0
                image = np.expand_dims(images, axis=0)
                labels = np.array(label)

                # lable = keras.utils.np_utils.to_categorical(lable)
                # lable = np.expand_dims(lable, axis=0)
                label = labels.reshape(1, 8)

                yield (image, label)

    def save(self, model):
        print('Model Saved.')
        model.save_weights(self.model_file)

    def load(self, model):
        print('Model Loaded.')
        model.load_weights(self.model_file)

    def predict(self, model, image):
        # 预测样本分类
        print(image.shape)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image.astype('float32')
        image = np.expand_dims(image, axis=0)

        # 归一化
        result = model.predict(image)

        print(result)
        return result