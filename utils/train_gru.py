# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:37:44 2018
@author: yy
"""
import os, sys
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback, EarlyStopping
from data_stream import *
# from keras.utils.visualize_util import plot
# from visual_callbacks import AccLossPlotter
# plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True, save_graph_path=sys.path[0])

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras import backend as K
from ocr_gru import get_gru_ctc_model
import os
from keras.models import Model
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
seq_len = 4

label_count = len(char_set) + 1
image_size = (100, 70)

IMAGE_HEIGHT = image_size[1]
IMAGE_WIDTH = image_size[0]


# CNN网络模型�?
class Training_Predict:
    def __init__(self):
        self.base_model = None
        self.ctc_model = None
        self.conv_shape = None

    # 建立模型
    def build_model(self):
      self.conv_shape, self.base_model, self.ctc_model = get_gru_ctc_model(image_size, seq_len, label_count)

    def predict(self):
        file_list = []
        captcha = CAPTCHA_creater()
        X, Y = captcha.get_batch(6000)
        X = np.array(X)
        X = np.transpose(X, [0, 2, 1, 3])
        X = self.normalization(X)
        Y = np.argmax(Y, 2)
        y_pred = self.base_model.predict(X)
        shape = y_pred[:, :, :].shape  # 2:
        out = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:,
              :seq_len]  # 2:
        print()
        out = out + np.ones(out.shape())


        # error_count = 0
        # for i in range(len(X)):
        #     print(file_list[i])
        #     str_src = str(os.path.split(file_list[i])[-1]).split('.')[0].split('_')[-1]
        #     print(out[i])
        #     str_out = ''.join([str(char_set[x]) for x in out[i] if x != -1])
        #     print(str_src, str_out)
        #     if str_src != str_out:
        #         error_count += 1
        #         print('This is a error image---------------------------:', error_count)

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_epoch_end(self, epoch, logs=None):
            self.ctc_model.save_weights('ctc_model.w')
            self.base_model.save_weights('base_model.w')
            self.test()

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

        # 训练模型
    def normalization(self, input):
        input = input.astype(np.float64)/255.0
        # mean = np.mean(input)
        # input[:,:,0] -= np.mean(input[:,:,0])
        # input[:,:,1] -= np.mean(input[:,:,1])
        # input[:,:,2] -= np.mean(input[:,:,2])
        return input


    def train(self, batch_size=32, nb_epoch=15, data_augmentation=False):

        # X = np.load('imgs/train/hardness0/imgs.npy')
        # from PIL import Image
        # im = Image.fromarray(X[0,:,:,:])
        # im.show()
        #
        # X = np.transpose(X, [0,2,1,3])
        # Y = np.load('imgs/train/hardness0/labels.npy')
        # Y = np.argmax(Y,2)
        # print(Y[0])
        #
        # X_val = np.load('imgs/val/hardness0/imgs.npy')
        # X_val = np.transpose(X_val, [0, 2, 1, 3])
        # Y_val = np.load('imgs/val/hardness0/labels.npy')
        # Y_val = np.argmax(Y_val, 2)
        captcha = CAPTCHA_creater()
        X, Y = captcha.get_batch(60000)
        X = np.array(X)
        X = np.transpose(X, [0, 2, 1, 3])
        X = self.normalization(X)
        Y = np.argmax(Y, 2)
        # X_val = self.normalization(X_val)
        print('train----------', X.shape, Y.shape)
        conv_shape = self.conv_shape

        maxin = 60000
        result = self.ctc_model.fit([X[:maxin], Y[:maxin], np.array(np.ones(len(X)) * int(conv_shape[1]))[:maxin],
                                     np.array(np.ones(len(X)) * seq_len)[:maxin]], Y[:maxin],
                                    batch_size=128,
                                    epochs=5,
                                    validation_split=0.2,
                                    callbacks=[EarlyStopping(patience=10)],  # checkpointer, history,history, plotter,
                                    # validation_data=([X[maxin:], Y[maxin:], np.array(np.ones(len(X))*int(conv_shape[1]))[maxin:], np.array(np.ones(len(X))*seq_len)[maxin:]], Y[maxin:]),
                                    )
        self.predict()

    MODEL_PATH = './model/gru.model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.base_model.save(file_path + 'base')
        self.ctc_model.save(file_path + 'ctc')

    def load_model(self, file_path=MODEL_PATH):
        self.base_model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.base_model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))


if __name__ == '__main__':
    # 训练模型，这段代码不用，注释    
    model = Training_Predict()

    model.build_model()
    model.train()
    #
    # model.save_model(file_path = './model/gur_ctc_model.h5')

    # model.load_model(file_path='./model/gur_ctc_model.h5base')
    # model.predict()
    #
    # #评估模型
    # model = Model()
    # model.load_model(file_path = './model/gur_ctc_model.h5')
    # model.evaluate(dataset)

