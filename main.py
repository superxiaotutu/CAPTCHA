import tensorflow as tf
from utils.data_stream import *

# config
diff = "easy"
image_height, image_width, image_channel = 10, 10, 3
NumCAPTCHA, NumAlb = 4, 26

# Ph
input_img = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
input_label = tf.placeholder(tf.float32, [None, NumCAPTCHA, NumAlb])

# DATA
data_stream = CAPTCHA_creater(image_height, image_width, diff)

# process

# train model
def CNN_model():
    pass


# train op

# main loop