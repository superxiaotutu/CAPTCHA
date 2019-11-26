import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
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

    input_tensor = Input((img_height, img_width, 3), name='input')
    x = input_tensor
    x = preprocess(x)

    x = Lambda(cnn_part)(x)
    # x = cnn_part(x)
    conv_shape = x.get_shape()
    print(conv_shape)
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
    x = Dense(label_count, kernel_initializer='he_normal', activation='softmax', name='output')(x)

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


def cnn_3layer(x):
    for i in range(3):
        x = Conv2D(32 * 2 ** i, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
        # x = Convolution2D(32*2**i, (3, 3), activation='relu')(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        return x

def cnn_resnet(x):
    # CNN part resnet
    with tf.variable_scope('mini-resnet'):
        x = _conv2d(x, 'outfirst', 3, 3, 4, 1)
        conv1 = _residual_block(x, 4, if_first=True, name='block1')
        conv2 = _residual_block(conv1, 8, name='block2')
        conv3 = _residual_block(conv2, 16, name='block3')
        conv4 = _residual_block(conv3, 32, name='block4')
        conv5 = _residual_block(conv4, 64, name='block5')
    return conv5

def cnn_part(x):
    # CNN part inception
    with tf.variable_scope('mini-inception'):
        x = _conv2d(x, 'cnn1', 3, 3, 4, 1)
        x = _batch_norm('bn1', x)
        x = _leaky_relu(x, 0.01)
        x = _max_pool(x, 2, 2)
        x = _conv2d(x, 'cnn2', 3, 4, 8, 1)
        x = _batch_norm('bn2', x)
        x = _leaky_relu(x, 0.01)
        x = _max_pool(x, 2, 2)

        block_1 = _inception_block(x, 8, 'block1')
        block_1 = _max_pool(block_1, 2, 2)

        block_2 = _inception_block(block_1, 120, 'block2a')
        block_2 = _max_pool(block_2, 2, 2)

        x = _conv2d(block_2, 'cnn3', 3, 120, 64, 1)
        x = _batch_norm('bn3', x)
        x = _leaky_relu(x, 0.01)
    return x

def _residual_block(input_layer, output_channel, if_first=False, name=None):
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    with tf.variable_scope('conv1_in_block'):
        if if_first:
            x = _conv2d(input_layer, 'cnn1' + name, 3, input_channel, output_channel, 1)
        else:
            x = _conv2d(input_layer, 'cnn1' + name, 3, input_channel, output_channel, stride)
            x = _batch_norm('bn1' + name, x)
            x = _leaky_relu(x, 0.01)

    with tf.variable_scope('conv2_in_block'):
        x = _conv2d(x, 'cnn2' + name, 3, output_channel, output_channel, 1)
        x = _batch_norm('bn2' + name, x)
        conv2 = _leaky_relu(x, 0.01)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.max_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output

def _inception_block(input, input_channel, name=None):
    branch_0 = _conv2d(input, name + 'Conv2d_0a_1x1', 1, input_channel, 32, 1)

    branch_1 = _conv2d(input, name + 'Conv2d_1a_1x1', 1, input_channel, 32, 1)
    branch_1 = _batch_norm(name + "branch_1_bn", branch_1)
    branch_1 = _leaky_relu(branch_1, 0.01)
    branch_1 = _conv2d(branch_1, name + 'Conv2d_1b_3x3', 3, 32, 48, 1)

    branch_2 = _conv2d(input, name + 'Conv2d_2a_1x1', 1, input_channel, 8, 1)
    branch_2 = _batch_norm(name + "branch_2_bn", branch_2)
    branch_2 = _leaky_relu(branch_2, 0.01)
    branch_2 = _conv2d(branch_2, name + 'Conv2d_2b_3x3', 3, 8, 24, 1)

    branch_3 = _max_pool(input, 3, 1)
    branch_3 = _conv2d(branch_3, name + 'Conv2d_3b_1x1', 1, input_channel, 16, 1)

    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    net = _batch_norm(name + "net_bn", net)
    net = _leaky_relu(net, 0.01)
    return net

def _conv2d(x, name, filter_size, in_channels, out_channels, strides):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable(name='W',
                                 shape=[filter_size, filter_size, in_channels, out_channels],
                                 dtype=tf.float32,
                                 initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer

        b = tf.get_variable(name='b',
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer())

        con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

    return tf.nn.bias_add(con2d_op, b)

def _batch_norm(name, x):
    """Batch normalization."""
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x_bn = tf.contrib.layers.batch_norm(
            inputs=x,
            decay=0.9,
            center=True,
            scale=True,
            epsilon=1e-5,
            updates_collections=None,
            is_training=True,
            fused=True,
            data_format='NHWC',
            zero_debias_moving_mean=True,
            scope='BatchNorm'
        )
    return x_bn

def _leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def _max_pool(x, ksize, strides):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, strides, strides, 1],
                          padding='SAME',
                          name='max_pool')

def _avg_pool(x, ksize, strides):
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, strides, strides, 1],
                          padding='SAME',
                          name='avg_pool')