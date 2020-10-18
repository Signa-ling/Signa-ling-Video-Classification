import h5py
import numpy as np

from sklearn.model_selection import train_test_split

from modules.utils import make_dir, plot_history
from model.create_model import ModelCreate


def model_train(X_train, y_train, depth, config_dict, flags):
    batch_size = config_dict['batch_size']
    epochs = config_dict['epoch']
    lr = config_dict['lr']
    num_classes = config_dict['num_classes']
    model_mode = config_dict['model_mode']
    img_shape = config_dict['img_shape']
    channel = 1

    if flags['color']:
        channel = 3

    print(f'ndim: {X_train.ndim}\
    \nshape: {X_train.shape}\
    \nimages: {len(X_train)}\
    \nlabels: {len(y_train)}')

    X_train = X_train.reshape((X_train.shape[0], depth, img_shape[0], img_shape[1], channel))
    X_train = X_train.astype("float32")
    # データのロード
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          test_size=0.2)

    print(X_train.ndim, X_train.shape, type(X_train))

    # モデル作成, model_modeに合わせて構築を変える
    model = ModelCreate(X_train, lr, num_classes)
    if model_mode == '3DCNN':
        model = model.CNN3D_model()
    elif model_mode == 'C3D':
        model = model.C3D_model()
    elif model_mode == 'LSTM':
        model = model.LSTM_model()

    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                        batch_size=batch_size, epochs=epochs, verbose=1)

    # 保存名
    save_name = 'Batch{0}_Epoch{1}_LR{2}'.format(batch_size, epochs,
                                                 lr)
    extends = ['.hdf5', '.h5', '_']
    save_name_list = [save_name + e for e in extends]

    # 保存先
    subs = ['./weight/', './model/', './Result/']
    path_sub = config_dict["version"] + '/' + model_mode + '/'
    save_path_list = [s + path_sub for s in subs]
    for path_name in save_path_list:
        make_dir(path_name)

    # モデルの重み、モデル本体を保存
    model.save_weights(save_path_list[0] + save_name_list[0])
    model.save(save_path_list[1] + save_name_list[1], include_optimizer=False)

    # 学習履歴をプロット
    plot_history(history, save_path_list[2], save_name_list[2])
