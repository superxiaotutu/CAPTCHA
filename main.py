import tensorflow as tf
from utils.data_stream import *
from utils.config import *
from utils.train_method import *

# Ph
input_img = tf.placeholder(tf.float32, [batchsize, image_width, image_height, image_channel])
input_label = tf.placeholder(tf.float32, [batchsize, NumCAPTCHA, NumAlb])
# input_label = tf.sparse_placeholder(tf.int32)
keep_prob = tf.placeholder(tf.float32)

# DATA
data_stream = CAPTCHA_creater()

# process


# train model
CNN_end_points = CTC_model(input_img, input_label, keep_prob)

# train op
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(CNN_end_points['loss'])
loss = CNN_end_points['loss']

# main loop
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = data_stream.get_batch(batchsize)
    t_loss, _ = sess.run([loss, train_op], feed_dict={input_img: batch[0], input_label: batch[1], keep_prob: 0.5})
    print(t_loss)
