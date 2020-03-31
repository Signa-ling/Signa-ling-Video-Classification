import numpy as np
import os

from tqdm import tqdm
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils

from utils import get_train_test_list, return_save_img_path
from video_to_frame import video_to_frame


def _test_labeling(data_list, class_dict):
    test_label = []

    for test_data in data_list:
        test_data = test_data.split('/')
        ind = class_dict[test_data[0]]
        test_label.append(int(ind)-1)

    return test_label


def _generator_img(image_shape, color_flag, imgs_path):
    height, width, depth = image_shape
    imgs = []
    for img_path in tqdm(imgs_path):
        frame_num = len(os.listdir(img_path))

        # depthより動画のフレーム数が少なければ使用するフレーム番号を水増し
        # そうでなければdepthの指定数だけ使用するフレーム番号を決める
        if frame_num < depth:
            dframe = depth - frame_num
            iframe = round(dframe / 2)
            fframe = dframe - iframe

            iframes = [1 for x in range(iframe)]
            nframes = [x+1 for x in range(frame_num)]
            fframes = [frame_num for x in range(fframe)]
            frames = iframes + nframes + fframes

        else:
            frames = [round(x * frame_num / depth + 1) for x in range(depth)]

        frame_array = []

        # frames内の番号と一致する画像をフォルダから探し出し読み込む
        for i in range(depth):
            image_name = os.path.join(img_path, str(frames[i])+'.jpg')
            if color_flag:
                img = load_img(image_name, target_size=(height, width))
            else:
                img = load_img(image_name, color_mode="grayscale", target_size=(height, width))

            if img is None:
                print("ERROR: can not read image : ", image_name)
            else:
                frame_array.append(img_to_array(img))

        imgs.append(frame_array)

    print(len(imgs))

    imgs = np.array(imgs)
    imgs = imgs.transpose((0, 2, 3, 1, 4))

    if not color_flag:
        imgs = imgs.astype('float32')
        imgs /= 255.0

    print(imgs.shape, imgs.ndim)

    return imgs


def load_data(image_shape, class_num, ver, use_mode, color_flag=False, frame_flag=True):
    data_list_path = "./ucfTrainTestlist/"
    root_path = './UCF-101/'
    save_root_path = './img/' + ver + '/'

    data_list, train_label, class_dict = get_train_test_list(data_list_path, root_path, class_num, ver)

    path_mode = use_mode

    if path_mode == 'train':
        data = data_list[0]
    else:
        data = data_list[1]

    if frame_flag:
        save_path = save_root_path + path_mode + '/'
        save_img_path_list = video_to_frame(root_path, save_path, data, ver)

    else:
        for i in range(2):
            save_img_path_list = []

            for file_name in data:
                save_path = save_root_path + path_mode + '/'
                _, save_img_path = return_save_img_path(file_name, save_path)
                save_img_path_list.append(save_img_path)

    if path_mode == 'train':
        X_train = _generator_img(image_shape, color_flag, save_img_path_list)
        y_train = np_utils.to_categorical(train_label, class_num)
        return X_train, y_train

    X_test = _generator_img(image_shape, color_flag, save_img_path_list)
    test_label = _test_labeling(data, class_dict)
    y_test = np_utils.to_categorical(test_label, class_num)
    return X_test, y_test
