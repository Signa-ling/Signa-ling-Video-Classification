import h5py
import numpy as np

from sklearn.model_selection import train_test_split

from utils import make_dir, load_json, plot_history
from load_data import load_data
from model import ModelCreate


def main():
    h_params, configs = load_json()
    print(configs)
    BATCH_SIZE, EPOCHS, NUM_CLASSES, LEARNING_RATE = h_params
    image_shape, load_ver, model_mode, use_mode, color_flag, frame_flag = configs

    # データのロード
    # 細かい処理はload_data.pyで行う
    X_train, y_train = load_data(image_shape, NUM_CLASSES, load_ver, use_mode,
                                 color_flag=color_flag, frame_flag=frame_flag)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          test_size=0.2)

    # モデル作成, model_modeに合わせて構築を変える
    model = ModelCreate(X_train, LEARNING_RATE, NUM_CLASSES)
    if model_mode == '3DCNN':
        model = model.CNN3D_model()
    elif model_mode == 'C3D':
        model = model.C3D_model()

    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                        batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

    # 保存名
    save_name = 'Batch{0}_Epoch{1}_LR{2}'.format(BATCH_SIZE, EPOCHS,
                                                 LEARNING_RATE)
    extends = ['.hdf5', '.h5', '_']
    save_name_list = [save_name + e for e in extends]

    # 保存先
    subs = ['./weight/', './model/', './Result/']
    path_sub = load_ver + '/' + model_mode + '/'
    save_path_list = [s + path_sub for s in subs]
    for path_name in save_path_list:
        make_dir(path_name)

    # モデルの重み、モデル本体を保存
    model.save_weights(save_path_list[0] + save_name_list[0])
    model.save(save_path_list[1] + save_name_list[1], include_optimizer=False)

    # 学習履歴をプロット
    plot_history(history, save_path_list[2], save_name_list[2])


if __name__ == "__main__":
    main()
