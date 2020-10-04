import h5py

from load_data import load_data
from model import ModelCreate
from utils import load_json


def main():
    h_params, configs = load_json()
    BATCH_SIZE, EPOCHS, NUM_CLASSES, LEARNING_RATE = h_params
    image_shape, load_ver, model_mode, use_mode, color_flag, frame_flag = configs

    weight_name = 'Batch{0}_Epoch{1}_LR{2}'.format(BATCH_SIZE, EPOCHS,
                                                   LEARNING_RATE)
    weights_path = './weight/{0}/{1}/{2}.hdf5'.format(load_ver, model_mode,
                                                      weight_name)

    X_test, y_test = load_data(image_shape, NUM_CLASSES, load_ver, use_mode,
                               color_flag=color_flag, frame_flag=frame_flag)

    model = ModelCreate(X_test, LEARNING_RATE, NUM_CLASSES)
    if model_mode == '3DCNN':
        model = model.CNN3D_model()
    elif model_mode == 'C3D':
        model = model.C3D_model()
    elif model_mode == 'LSTM':
        model = model.LSTM_model()
        X_test = X_test.transpose((0, 3, 1, 2, 4))

    model.load_weights(weights_path)
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


if __name__ == "__main__":
    main()
