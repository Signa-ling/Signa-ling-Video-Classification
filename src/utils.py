import json
import os
import matplotlib.pyplot as plt


def _read_data_class_list(txt_file, use_class, mode='train'):
    data_list = []
    class_ind = []

    with open(txt_file) as txtlist:
        while True:
            txtlist_line = txtlist.readline()
            if not txtlist_line:
                break

            if mode == 'train':
                txtlist_line = txtlist_line.split(' ')
                if txtlist_line[0].split('/')[0] in use_class:
                    data_list.append(txtlist_line[0])
                    class_ind.append(int(txtlist_line[1][:-1])-1)

            elif mode == 'test':
                if txtlist_line.split('/')[0] in use_class:
                    data_list.append(txtlist_line[:-1])

            else:
                txtlist_line = txtlist_line.split(' ')
                if txtlist_line[1][:-1] in use_class:
                    class_ind.append(txtlist_line[0])
                    data_list.append(txtlist_line[1][:-1])

    if mode != 'test':
        return data_list, class_ind

    return data_list


def get_train_test_list(data_list_path, root_path, class_num, ver):
    use_class = os.listdir(root_path)[:class_num]
    train_txt = data_list_path + "trainlist" + ver + ".txt"
    test_txt = data_list_path + "testlist" + ver + ".txt"
    class_txt = data_list_path + 'classInd.txt'
    train_list, train_label = _read_data_class_list(train_txt, use_class)
    test_list = _read_data_class_list(test_txt, use_class, mode='test')
    class_list, class_ind = _read_data_class_list(class_txt, use_class, mode='class')
    class_dict = dict(zip(class_list, class_ind))
    data_list = (train_list, test_list)
    return data_list, train_label, class_dict


def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print("Make Directory {0}".format(dir_path))


def return_save_img_path(file_name, save_path):
    file_name_split = os.path.split(file_name)
    save_dir_name = file_name_split[0]
    save_img_name = file_name_split[1].split('.')

    save_dir_path = save_path + save_dir_name
    save_img_path = save_dir_path + '/' + save_img_name[0] + '/'

    return save_dir_path, save_img_path


def load_json():
    with open('./config.json', 'r') as json_file:
        json_data = json.load(json_file)

    # 各パラメータ, 設定
    BATCH_SIZE = json_data['HYPER_PARAMETER']['BATCH_SIZE']
    EPOCHS = json_data['HYPER_PARAMETER']['EPOCHS']
    NUM_CLASSES = json_data['HYPER_PARAMETER']['NUM_CLASSES']
    LEARNING_RATE = json_data['HYPER_PARAMETER']['LEARNING_RATE']
    h_params = [BATCH_SIZE, EPOCHS, NUM_CLASSES, LEARNING_RATE]

    image_shape = tuple(json_data['CONFIG']['IMG_SHAPE'])
    load_ver = json_data['CONFIG']['LOAD_VERSION']['01']
    model_mode = json_data['CONFIG']['MODEL_MODE']['LSTM']
    use_mode = json_data['CONFIG']['USE_MODE']['TRAIN']
    color_flag = json_data['CONFIG']['COLOR_FLAG']
    frame_flag = json_data['CONFIG']['FRAME_FLAG']
    configs = [image_shape, load_ver, model_mode, use_mode,
               color_flag, frame_flag]

    return h_params, configs


def plot_history(history, result_path, result_name):
    plt_acc_path = os.path.join(result_path, result_name + 'acc.png')
    plt_loss_path = os.path.join(result_path, result_name + 'loss.png')

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(plt_acc_path)

    plt.show()

    # 損失の履歴をプロット

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.savefig(plt_loss_path)
    plt.show()
