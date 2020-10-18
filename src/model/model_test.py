import h5py
import numpy as np
import os

from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import Adam

from modules.load_data import load_data
from model.create_model import ModelCreate


def model_test(X_test, y_test, depth, config_dict, flags):
    lr = config_dict['lr']
    num_classes = config_dict['num_classes']
    model_mode = config_dict['model_mode']
    img_shape = config_dict['img_shape']
    version = config_dict['version']
    channel = 1

    if flags['color']:
        channel = 3

    print(f'ndim: {X_test.ndim}\
        \nshape: {X_test.shape}\
        \nimages: {len(X_test)}\
        \nlabels: {len(y_test)}')

    X_test = X_test.reshape((X_test.shape[0], depth, img_shape[0], img_shape[1], channel))
    X_test = X_test.astype("float32")

    # y_test = np_utils.to_categorical(y_test, num_classes)
    print(X_test.ndim, X_test.shape)

    model_save_folder = f'./model/{version}/{model_mode}'
    model_name = input('What is model name?: ') + '.h5'
    model_path = os.path.join(model_save_folder, model_name)
    print(model_path)

    if model_name in os.listdir(model_save_folder):
        model = load_model(model_path, compile=False)
    else:
        return print('Model load Error')

    opt = Adam(lr=lr)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['acc'])
    model.summary()

    expect = np.argmax(model.predict(X_test), axis=-1)
    print("expect ok")
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
