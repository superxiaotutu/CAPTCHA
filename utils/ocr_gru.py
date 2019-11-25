import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
import keras
from keras.models import Model
from keras import optimizers
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate, Flatten, \
    Dense, Dropout, GRU, LSTM, Add
from keras.regularizers import l2
import keras.backend as K

char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
image_size = (100, 70)

IMAGE_HEIGHT = image_size[1]
IMAGE_WIDTH = image_size[0]


def get_gru_ctc_model(image_size=image_size,
                      seq_len=4,  # 字符最大长
                      label_count=37):  # 标签数量
    img_height, img_width = image_size[0], image_size[1]

    input_tensor = Input((img_height, img_width, 3),name='input')
    x = input_tensor
    x = preprocess(x)
    x = cnn_part(x)
    conv_shape = x.get_shape()
    # print(conv_shape)
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

    x = Dense(32, activation='relu')(x)

    gru_1 = GRU(32, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
    gru1_merged = Add()([gru_1, gru_1b])

    gru_2 = GRU(32, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)

    x = Concatenate()([gru_2, gru_2b])
    x = Dropout(0.25)(x)
    x = Dense(label_count, kernel_initializer='he_normal', activation='softmax',name='output')(x)
    base_model = Model(inputs=input_tensor, outputs=x)

    labels = Input(name='the_labels', shape=[seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    ctc_model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
    sgd = optimizers.Adam(lr=0.005)
    ctc_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    ctc_model.summary()
    return conv_shape, base_model, ctc_model


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def preprocess(input):

    output = input
    return output

def cnn_part(x):
    for i in range(3):
        x = Conv2D(32 * 2 ** i, (3, 3), activation='relu', strides=(2,2),padding='same')(x)
        # x = Convolution2D(32*2**i, (3, 3), activation='relu')(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        return x