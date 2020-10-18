import argparse
import json

from modules.load_data import load_data
from model import model_train, model_test


with open('./config.json', 'r') as config_json:
    json_data = json.load(config_json)

BATCH_SIZE = json_data['HYPER_PARAMETER']['BATCH_SIZE']
EPOCHS = json_data['HYPER_PARAMETER']['EPOCHS']
LEARNING_RATE = json_data['HYPER_PARAMETER']['LEARNING_RATE']
NUM_CLASSES = json_data['HYPER_PARAMETER']['NUM_CLASSES']
IMAGE_SHAPE = json_data['CONFIG']['IMG_SHAPE']

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', default=BATCH_SIZE, type=int)
parser.add_argument('-e', '--epoch', default=EPOCHS, type=int)
parser.add_argument('-l', '--learningrate', default=LEARNING_RATE, type=float)
parser.add_argument('-v', '--version', default='01', choices=['02', '03'])
parser.add_argument('-n', '--numclasses', default=NUM_CLASSES, type=int)
parser.add_argument('-m', '--model', default='3DCNN', choices=['C3D', 'LSTM'])
parser.add_argument('-ff', '--frameflag', action='store_true',
                    help='true: video→frame image create, default: false')
parser.add_argument('-mf', '--modeflag', action='store_false',
                    help='false: test mode, default: true')
parser.add_argument('-cf', '--colorflag', action='store_true',
                    help='false: monochrome mode, default: false')
args = parser.parse_args()


def main():
    flags = {"frame": args.frameflag, "mode": args.modeflag,
             "color": args.colorflag}
    config_dict = {"batch_size": args.batch, "epoch": args.epoch,
                   "lr": args.learningrate, "num_classes": args.numclasses,
                   "model_mode": args.model, "version": args.version,
                   "img_shape": IMAGE_SHAPE}
    X, labels, depth = load_data(flags, config_dict)
    if flags["mode"]:
        model_train.model_train(X, labels, depth, config_dict, flags)
    else:
        model_test.model_test(X, labels, depth, config_dict, flags)


if __name__ == "__main__":
    main()
