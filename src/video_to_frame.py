import cv2
import numpy as np

from utils import make_dir, return_save_img_path


def video_to_frame(root_path, save_path, data_list, ver):
    save_img_path_list = []
    make_dir(save_path)

    for file_name in data_list:
        save_dir_path, save_img_path = return_save_img_path(file_name, save_path)
        make_dir(save_dir_path)
        make_dir(save_img_path)
        save_img_path_list.append(save_img_path)

        video_path = root_path + file_name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            break

        n = 1

        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('{}{}.{}'.format(save_img_path, n, 'jpg'), frame)
                n += 1
            else:
                break

    return save_img_path_list
