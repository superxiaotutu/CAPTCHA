import cv2
import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageFilter
from captcha.image import DEFAULT_FONTS, ImageCaptcha
from skimage.util import random_noise
import matplotlib.pyplot as plt

def head_B(self, input, phi=0.8):
        kernel_3_1 = np.ones([1, 1, 3, 1]) / 3
        kernel_1_3 = np.ones([1, 1, 1, 3])
        a = tf.nn.conv2d(input, kernel_3_1, strides=[1, 1, 1, 1], padding="SAME")
        b = tf.nn.conv2d(a, kernel_1_3, strides=[1, 1, 1, 1], padding="SAME")
        S = 1 / (1 + tf.exp(- 20 * (b - phi)))
        return S

    def head_Guss(self, inputs, kerStd=0.8):
        def getGuessValue(kerStd, posX, posY):
            return 1. / (2. * np.pi * (np.power(kerStd, 2))) * np.exp(
                -(np.power(posX, 2) + np.power(posY, 2)) / (2. * (np.power(kerStd, 2))))

        def getGuessKernel(kerStd):
            K11 = np.eye(3) * getGuessValue(kerStd, -1, 1)
            K12 = np.eye(3) * getGuessValue(kerStd, 0, 1)
            K13 = np.eye(3) * getGuessValue(kerStd, 1, 1)
            K21 = np.eye(3) * getGuessValue(kerStd, -1, 0)
            K22 = np.eye(3) * getGuessValue(kerStd, 0, 0)
            K23 = np.eye(3) * getGuessValue(kerStd, 1, 0)
            K31 = np.eye(3) * getGuessValue(kerStd, -1, -1)
            K32 = np.eye(3) * getGuessValue(kerStd, 0, -1)
            K33 = np.eye(3) * getGuessValue(kerStd, 1, -1)
            kernel = tf.constant(np.array([[K11, K12, K13], [K21, K22, K23], [K31, K32, K33]]),
                                 dtype=tf.float32)  # 3*3*4*4
            return kernel

        kernel = getGuessKernel(kerStd)
        return tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding="SAME")
