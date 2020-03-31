from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, ZeroPadding3D)
from keras.models import Sequential
from keras.optimizers import Adam


class model_create():
    def __init__(self, shape, lr, num_classes):
        self.shape = shape
        self.lr = lr
        self.num_classes = num_classes

    def CNN3D_model(self):
        model = Sequential()
        model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
                  self.shape.shape[1:]), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
        model.add(Activation('softmax'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
        model.add(Dropout(0.25))

        model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
        model.add(Activation('softmax'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        opt = Adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['acc'])
        model.summary()

        return model

    def C3D_model(self):
        model = Sequential()

        # 1st layer group
        model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu',
                         padding='same', name='conv1',
                         strides=(1, 1, 1), input_shape=self.shape.shape[1:]))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               padding='valid', name='pool1'))

        # 2nd layer group
        model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu',
                         padding='same', name='conv2', strides=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               padding='valid', name='pool2'))

        # 3rd layer group
        model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu',
                         padding='same', name='conv3a',
                         strides=(1, 1, 1)))
        model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu',
                         padding='same', name='conv3b',
                         strides=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               padding='valid', name='pool3'))

        # 4th layer group
        model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu',
                         padding='same', name='conv4a',
                         strides=(1, 1, 1)))
        model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu',
                         padding='same', name='conv4b',
                         strides=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               padding='valid', name='pool4'))

        # 5th layer group
        model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu',
                         padding='same', name='conv5a',
                         strides=(1, 1, 1)))
        model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu',
                         padding='same', name='conv5b',
                         strides=(1, 1, 1)))
        model.add(ZeroPadding3D(padding=(0, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               padding='valid', name='pool5'))
        model.add(Flatten())

        # FC layers group
        model.add(Dense(4096, activation='relu', name='fc6'))
        model.add(Dropout(.5))
        model.add(Dense(4096, activation='relu', name='fc7'))
        model.add(Dropout(.5))
        model.add(Dense(self.num_classes, activation='softmax', name='fc8'))

        opt = Adam(lr=self.lr)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['acc'])
        model.summary()

        return model
