# coding=utf-8
import model_cnn

# def data_label(path):
#     f = open(path + "lable-40.txt", "r")
#     j = 0
#     i = -1
#     datalist = []
#     labellist = []
#     while True:
#
#         for line in f.readlines():
#             i += 1
#             j += 1
#             a = line.replace("\n", "")
#             b = a.split(",")
#             lable = b[1:]
#             # print(b[1:])
#             #对标签进行归一化（不归一化也行）
#             # for num in b[1:]:
#             #     lab = int(num) / 255.0
#             #     labellist.append(lab)
#             # lab = labellist[i * 10:j * 10]
#             imgname = path + b[0]
#             images = load_img(imgname)
#             images = img_to_array(images).astype('float32')
#             # 对图片进行归一化（不归一化也行）
#             # images /= 255.0
#             image = np.expand_dims(images, axis=0)
#             lables = np.array(lable)
#
#             # lable =keras.utils.np_utils.to_categorical(lable)
#             # lable = np.expand_dims(lable, axis=0)
#             lable = lables.reshape(1, 10)
#
#             yield (image,lable)


###############
# 开始建立CNN模型
###############
# 生成一个model
# class Model(object):
#     def __init__(self, img_size=178, batch_size=32):
#         self.img_size = img_size
#         self.batch_size = batch_size
#         pass
#
#     def get_cnn_model(self):
#         model = Sequential()            # 178*178*3
#         model.add(Conv2D(32, (3, 3), input_shape=(self.img_size, self.img_size, 3)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#
#         model.add(Conv2D(32, (3, 3)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#
#         model.add(Conv2D(64, (3, 3)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#
#         model.add(Flatten())
#         model.add(Dense(64))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(8))
#         model.summary()
#         return model
#
#     def train(self, model, epoch_num):
#         # print(lable.shape)
#         model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#         # optimizer = SGD(lr=0.03, momentum=0.9, nesterov=True)
#         # model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
#         # epoch_num = 10
#
#         learning_rate = np.linspace(0.03, 0.01, epoch_num)
#         change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
#         early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
#         check_point = ModelCheckpoint('CNN_model_final.h5', monitor='val_loss', verbose=0, save_best_only=True,
#                                       save_weights_only=False, mode='auto', period=1)
#
#         model.fit_generator(data_label(train_path), callbacks=[check_point, early_stop, change_lr],
#                             samples_per_epoch=int(train_samples // self.batch_size), epochs=epoch_num,
#                             validation_steps=int(test_samples // self.batch_size), validation_data=data_label(test_samples))
#
#         # model.fit(traindata, trainlabel, batch_size=32, epochs=50,
#         #           validation_data=(testdata, testlabel))
#         model.evaluate_generator(data_label(test_samples))
#
#     def save(self, model, file_path=FILE_PATH):
#         print('Model Saved.')
#         model.save_weights(file_path)
#
#     def load(self, model, file_path=FILE_PATH):
#         print('Model Loaded.')
#         model.load_weights(file_path)
#
#     def predict(self, model, image):
#         # 预测样本分类
#         print(image.shape)
#         image = cv2.resize(image, (self.img_size, self.img_size))
#         image.astype('float32')
#         image = np.expand_dims(image, axis=0)
#
#         # 归一化
#         result = model.predict(image)
#
#         print(result)
#         return result


if __name__ == '__main__':
    FILE_PATH = './CNN_model_final.h5'
    train_path = '/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_train'
    test_path = '/home/caiyueliang/deeplearning/CarPlatePointsDetect/Data/car_plate_test'

    models = model_cnn.ModelCNN(train_path, test_path, FILE_PATH)

    my_model = models.get_model()
    # models.load(my_model)
    models.train(my_model)
    models.save(my_model)
