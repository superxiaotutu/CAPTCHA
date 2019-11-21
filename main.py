import tensorflow as tf
from utils.data_stream import *
from utils.config import *
from utils.train_method import *

# Ph
input_img = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
input_label = tf.placeholder(tf.float32, [None, NumCAPTCHA, NumAlb])
keep_prob = tf.placeholder(tf.float32)

# DATA
data_stream = CAPTCHA_creater(image_height, image_width, diff)

# process


# train model
CNN_end_points = CNN_model(input_img, input_label, keep_prob)

# train op


# main loop

