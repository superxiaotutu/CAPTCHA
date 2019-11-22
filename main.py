import tensorflow as tf
from utils.data_stream import *
from utils.config import *
from utils.train_method import *

# Ph
input_img = tf.placeholder(tf.float32, [None, image_width, image_height, image_channel])
input_label = tf.placeholder(tf.float32, [None, NumCAPTCHA, NumAlb])
keep_prob = tf.placeholder(tf.float32)

# DATA
data_stream = CAPTCHA_creater()

# process


# train model
CNN_end_points = CNN_model(input_img, input_label, keep_prob)

# train op
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(CNN_end_points['loss'])
loss = CNN_end_points['loss']

# main loop
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = data_stream.get_batch(100)
    t_loss, _ = sess.run([loss, train_op], feed_dict={input_img: batch[0], input_label: batch[1], keep_prob: 0.5})
    print(t_loss)
