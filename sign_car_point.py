# coding=utf-8
import cv2
import os
import time

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
def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


class SignCarPoint:
    def __init__(self, image_dir, label_file, index_file):
        self.img_files = common.get_files(image_dir)
        self.image_dir = image_dir
        self.label_file = label_file
        self.car_points = []
        self.index_file = index_file
        return

    def mouse_click_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.car_points) < 4:
                cv2.circle(self.img, (x, y), 3, (0, 0, 255), -1)
                print('click: [%d, %d]' % (x, y))
                self.car_points.append((x, y))
            else:
                print('self.car_points is too long, %s' % str(self.car_points))

    def sign_start(self, restart=False):
        times = 2

        cv2.namedWindow('sign_image')
        cv2.setMouseCallback('sign_image', self.mouse_click_events)    # 鼠标事件绑定

        if restart is False:
            try:
                start_i = int(common.read_data(self.index_file, 'r'))
                print('start_index: ' + str(start_i))
            except Exception, e:
                print e
                start_i = 0
        else:
            start_i = 0

        # for img_file in self.img_files:
        while start_i < len(self.img_files):
            print('[total] %d; [index] %d; [name] %s' % (len(self.img_files), start_i, self.img_files[start_i]))

            self.img = cv2.imread(self.img_files[start_i])
            self.img = cv2.resize(self.img, (self.img.shape[0]*times, self.img.shape[1]*times))

            while True:
                cv2.imshow('sign_image', self.img)

                # 保存这张图片
                k = cv2.waitKey(1) & 0xFF
                if k == ord('s'):
                    print('save ...')
                    data = self.img_files[start_i] + " " + str(len(self.car_points))
                    for (x, y) in self.car_points:
                        data += ' ' + str(x/float(times)) + ' ' + str(y/float(times))
                    data += '\n'

                    common.write_data(self.label_file, data, 'a+')
                    start_i += 1
                    common.write_data(self.index_file, str(start_i), 'w')
                    self.car_points = []
                    break

                if k == ord('d'):
                    print('delete ...')
                    common.exe_cmd('rm -r ' + self.img_files[start_i])
                    self.img_files.pop(start_i)

                    self.img = cv2.imread(self.img_files[start_i])
                    self.img = cv2.resize(self.img, (self.img.shape[0] * times, self.img.shape[1] * times))
                    cv2.imshow('sign_image', self.img)
                    self.car_points = []

                # 重新加载图片
                if k == ord('r'):
                    print('re sign ...')
                    self.img = cv2.imread(self.img_files[start_i])
                    self.img = cv2.resize(self.img, (self.img.shape[0] * times, self.img.shape[1] * times))
                    cv2.imshow('sign_image', self.img)
                    self.car_points = []

                if k == ord('c'):
                    print('change size ...')
                    if times == 2:
                        times = 4
                    else:
                        times = 2
                    self.img = cv2.imread(self.img_files[start_i])
                    self.img = cv2.resize(self.img, (self.img.shape[0] * times, self.img.shape[1] * times))
                    cv2.imshow('sign_image', self.img)
                    self.car_points = []

    def check_start(self):
        times = 2

        with open(self.label_file) as f:
            for line in f.readlines():
                line = line.replace('\r', '').replace('\n', '')
                print(line)
                list_str = line.split(' ')

                self.img = cv2.imread(list_str[0])
                self.img = cv2.resize(self.img, (self.img.shape[0] * times, self.img.shape[1] * times))

                if int(list_str[1]) != 4:
                    print('[ERROR] ' + list_str[0] + ' points not 4 !!')

                # cv2.circle(self.img, (x, y), 3, (0, 0, 255), -1)
                # cv2.circle(self.img, (int((float(list_str[2])*times), int(float(list_str[3])*times))), 3, (0, 0, 255), -1)

                cv2.circle(self.img, (int(float(list_str[2])*times), int(float(list_str[3])*times)), 3, (0, 0, 255), -1)
                cv2.line(self.img, (int(float(list_str[2])*times), int(float(list_str[3])*times)),
                         (int(float(list_str[4]) * times), int(float(list_str[5]) * times)), (0, 0, 255), 3, -1)

                cv2.circle(self.img, (int(float(list_str[4])*times), int(float(list_str[5])*times)), 3, (0, 255, 255), -1)
                cv2.line(self.img, (int(float(list_str[4])*times), int(float(list_str[5])*times)),
                         (int(float(list_str[6]) * times), int(float(list_str[7]) * times)), (0, 255, 255), 3, -1)

                cv2.circle(self.img, (int(float(list_str[6])*times), int(float(list_str[7])*times)), 3, (255, 0, 0), -1)
                cv2.line(self.img, (int(float(list_str[6])*times), int(float(list_str[7])*times)),
                         (int(float(list_str[8]) * times), int(float(list_str[9]) * times)), (255, 0, 0), 3, -1)

                cv2.circle(self.img, (int(float(list_str[8])*times), int(float(list_str[9])*times)), 3, (0, 255, 0), -1)

                cv2.imshow('check_image', self.img)
                cv2.waitKey(0)

    def change_size(self, root_path, label_file, output_path):
        times = 1
        ratio = 0.2

        with open(label_file) as f:
            for line in f.readlines():
                line = line.replace('\r', '').replace('\n', '')
                print(line)
                list_str = line.split(' ')

                image_path = os.path.join(root_path, list_str[0])
                print(image_path)

                img = cv2.imread(image_path)
                h, w, c = img.shape
                print(h, w, c)
                # img = cv2.resize(img, (img.shape[0] * times, img.shape[1] * times))

                padding_h = int(h * ratio)
                # image[padding:resize_h-padding, 0:image.shape[1]]
                new_img = img[padding_h:h-padding_h, 0:w].copy()

                if int(list_str[1]) != 4:
                    print('[ERROR] ' + list_str[0] + ' points not 4 !!')

                cv2.circle(img, (int(float(list_str[2])*times), int(float(list_str[3]))*times), 3, (0, 0, 255), -1)
                cv2.circle(img, (int(float(list_str[4])*times), int(float(list_str[5]))*times), 3, (255, 255, 0), -1)
                cv2.circle(img, (int(float(list_str[6])*times), int(float(list_str[7]))*times), 3, (255, 0, 0), -1)
                cv2.circle(img, (int(float(list_str[8])*times), int(float(list_str[9]))*times), 3, (0, 255, 0), -1)
                cv2.imshow('check_image', img)

                point_list = list()
                for i in range(2, 10):
                    if i % 2 != 0:
                        point_list.append(float('%.2f' % ((float(list_str[i])-padding_h)*h/new_img.shape[0])))
                    else:
                        point_list.append(float(list_str[i]))

                print(point_list)
                new_img = cv2.resize(new_img, (h, w))

                my_img = new_img.copy()
                cv2.circle(my_img, (int(float(point_list[0]) * times), int(float(point_list[1])) * times), 3, (0, 0, 255), -1)
                cv2.circle(my_img, (int(float(point_list[2]) * times), int(float(point_list[3])) * times), 3, (255, 255, 0), -1)
                cv2.circle(my_img, (int(float(point_list[4]) * times), int(float(point_list[5])) * times), 3, (255, 0, 0), -1)
                cv2.circle(my_img, (int(float(point_list[6]) * times), int(float(point_list[7])) * times), 3, (0, 255, 0), -1)
                cv2.imshow('new_image', my_img)

                while True:
                    # 保存这张图片
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('y'):
                        save_path = os.path.join(output_path, list_str[0])
                        print(save_path)
                        (file_path, tempfilename) = os.path.split(save_path)
                        print(file_path)
                        mkdir_if_not_exist(file_path)

                        cv2.imwrite(save_path, new_img)

                        data = list_str[0] + " " + str(len(point_list)/2)
                        for point in point_list:
                            data += ' ' + str(point)
                        data += '\n'

                        common.write_data(os.path.join(output_path, 'label.txt'), data, 'a+')

                        break
                    if k == ord('n'):
                        break

                print("====================================================================")

        return


if __name__ == '__main__':
    # image_dir = "../Data/car_finemap_detect/car_plate_test/failed_1"

    # image_dir = "../Data/car_finemap_detect/car_plate_train/data_2"
    # image_dir = "../Data/car_finemap_detect/car_plate_train/data_3"
    # image_dir = "../Data/car_finemap_detect/car_plate_train/szlg_1"
    # image_dir = "../Data/car_finemap_detect/car_plate_train/failed_1"
    image_dir = "../Data/car_finemap_detect/car_plate_train/failed_4"

    label_file = "./label.txt"
    index_file = "./index.txt"
    sign_point = SignCarPoint(image_dir, label_file, index_file)

    # sign_point.sign_start()

    # sign_point.check_start()

    # sign_point.change_size('../Data/car_finemap_detect/car_plate_train/',
    #                        '../Data/car_finemap_detect/car_plate_train/label.txt',
    #                        '../Data/car_finemap_detect_new/car_plate_train/')

    sign_point.change_size('../Data/car_finemap_detect/car_plate_test/',
                           '../Data/car_finemap_detect/car_plate_test/label.txt',
                           '../Data/car_finemap_detect_new/car_plate_test/')
