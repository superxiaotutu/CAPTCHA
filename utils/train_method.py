from utils.config import *
import tensorflow as tf

slim = tf.contrib.slim
def CNN_model(img_ph, la_ph, dropout_keep_prob):
    end_points = {}
    with tf.variable_scope('CifarNet'):
        net = slim.conv2d(img_ph, 64, [5, 5], scope='conv1')
        end_points['conv1'] = net
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        end_points['pool1'] = net
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        end_points['conv2'] = net
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        end_points['pool2'] = net
        net = slim.flatten(net)
        end_points['Flatten'] = net
        net = slim.fully_connected(net, 384, scope='fc3')
        end_points['fc3'] = net
        net = slim.dropout(net, dropout_keep_prob, is_training=True, scope='dropout3')
        net = slim.fully_connected(net, 192, scope='fc4')
        end_points['fc4'] = net

        logits = [slim.fully_connected(net, NumAlb, activation_fn=None, scope='logits') for i in range(NumCAPTCHA)]

        end_points['Logits'] = logits
        end_points['Predictions'] = [tf.nn.softmax(i) for i in logits]

    end_points['loss'] = tf.add_n([tf.losses.softmax_cross_entropy(la_ph[:, i, :], logits[i]) for i in range(4)])
    end_points['decoder'] = [tf.argmax(logits[i]) for i in range(4)]
    return end_points
