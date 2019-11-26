from utils.config import *
import tensorflow as tf

slim = tf.contrib.slim


def CNN_model(img_ph, la_ph, dropout_keep_prob):
    end_points = {}
    img_ph = img_ph / 255
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

        logits = [slim.fully_connected(net, NumAlb, activation_fn=None, scope='logits_' + str(i)) for i in
                  range(NumCAPTCHA)]

        end_points['Logits'] = logits
        end_points['Predictions'] = [tf.nn.softmax(i) for i in logits]

    end_points['loss'] = tf.add_n([tf.losses.softmax_cross_entropy(la_ph[:, i, :], logits[i]) for i in range(4)])
    end_points['decoder'] = [tf.argmax(logits[i]) for i in range(4)]
    return end_points


def CTC_model(img_ph, la_ph, dropout_keep_prob, mode='train'):
    end_point = {}
    # CNN part
    with tf.variable_scope('cnn'):
        x = img_ph
        outputs_size = [64, 128, 128, 64]
        for i in range(4):
            with tf.variable_scope('unit-%d' % (i + 1)):
                x = slim.conv2d(x, outputs_size[i], 3, normalizer_fn=slim.batch_norm, padding='SAME',
                                activation_fn=tf.nn.relu)
                x = slim.max_pool2d(x, 2, padding='SAME')
            _, feature_h, feature_w, _ = x.get_shape().as_list()

    # LSTM part
    with tf.variable_scope('lstm'):
        x = tf.transpose(x, [0, 2, 1, 3])  # [batch_size, feature_w, feature_h, FLAGS.out_channels]
        x = tf.reshape(x, [batchsize, feature_w, feature_h * 64])
        seq_len = tf.fill([x.get_shape().as_list()[0]], feature_w)
        cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
        if mode == 'train':
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=dropout_keep_prob)
        cell1 = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=True)
        if mode == 'train':
            cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=dropout_keep_prob)
        # Stacking rnn cells
        stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
        initial_state = stack.zero_state(batchsize, dtype=tf.float32)
        # The second output is the last state and we will not use that
        outputs, _ = tf.nn.dynamic_rnn(cell=stack, inputs=x, sequence_length=seq_len, initial_state=initial_state,
                                       dtype=tf.float32,
                                       time_major=False)  # [batch_size, max_stepsize, FLAGS.num_hidden]
        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, 128])  # [batch_size * max_stepsize, FLAGS.num_hidden]
        W = tf.get_variable(name='W_out', shape=[128, NumAlb], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
        b = tf.get_variable(name='b_out', shape=[NumAlb], dtype=tf.float32, initializer=tf.constant_initializer())
        logits = tf.matmul(outputs, W) + b
        # Reshaping back to the original shape
        shape = tf.shape(x)
        logits = tf.reshape(logits, [shape[0], -1, NumAlb])
        # Time major
        logits = tf.transpose(logits, (1, 0, 2))

    arr_tensor = la_ph
    arr_idx = tf.where(arr_tensor)
    arr_shape = arr_tensor.get_shape()
    arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(arr_tensor, arr_idx), arr_shape)
    arr_sparse = tf.cast(arr_sparse, tf.int32)
    loss = tf.nn.ctc_loss(labels=arr_sparse, inputs=logits, sequence_length=seq_len)
    cost = end_point['loss'] = tf.reduce_mean(loss)

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    dense_decoded = end_point['decoder'] = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
    return end_point
