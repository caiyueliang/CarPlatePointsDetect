# coding=utf-8
import numpy as np
import cv2
import os
import time
import training_judging as tj
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, BatchNormalization
from keras.optimizers import SGD

import keras
import numpy as np

import common as common

# ['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 'EVENT_FLAG_RBUTTON',
# 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK',
# 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL',
# 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']

# events = [i for i in dir(cv2) if 'EVENT' in i]
# img = np.zeros((512, 512, 3), np.uint8)


# mouse callback function
# def mouse_click_events(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img, (x, y), 100, (255, 0, 0), -1)


class SignCarPoint:
    def __init__(self, image_dir, label_file):
        self.img_files = common.get_files(image_dir)
        self.image_dir = image_dir
        self.label_file = label_file
        self.car_points = []
        pass

    def mouse_click_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.car_points) < 4:
                cv2.circle(self.img, (x, y), 3, (0, 0, 255), -1)
                print('click: [%d, %d]' % (x, y))
                self.car_points.append((x, y))
            else:
                print('self.car_points is too long, %s' % str(self.car_points))

    def sign_start(self):
        cv2.namedWindow('sign_image')
        cv2.setMouseCallback('sign_image', self.mouse_click_events)    # 鼠标事件绑定

        for img_file in self.img_files:
            self.img = cv2.imread(img_file)
            cv2.imshow('sign_image', self.img)

            while (True):
                cv2.imshow('sign_image', self.img)

                # 保存这张图片
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    
                    break

                # 重新加载图片
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    cv2.imshow('sign_image', self.img)
                    self.car_points = []

            # 保存点信息：
            self.car_points = []


if __name__ == '__main__':
    image_dir = "/cyl_data/car_plate"
    label_file = "/cyl_data/car_plate_label.txt"
    sign_point = SignCarPoint(image_dir, label_file)

    sign_point.sign_start()
