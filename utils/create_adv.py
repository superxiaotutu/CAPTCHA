import tensorflow as tf
from data_stream import *
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import keras
from train_gru import Training_Predict

class Adversarial_captcha:
    def __init__(self, input, output, lamda, alpha,label):
        self.input = input[0]
        self.output = output[0]
        self.lamda = lamda
        self.alpha = int(alpha)
        self.label = label

    # def get_target_laebl(self,):
    #     ori_label = tf.argmax(self.output,1)
    #     temp = [1 for i in range(ori_label.shape[0].value)]
    #     for i in range(ori_label.shape[0].value):
    #         if ori_label[i] == 35:
    #             temp[i] = -35
    #         elif ori_label[i] == 36:
    #             temp[i] = 0
    #     temp = tf.constant(temp,dtype=tf.int64)
    #     target_label = temp + ori_label
    #     return target_label

    def attack(self, input_adv):
        # input_adv = tf.cast(input_adv, dtype=tf.float32)
        loss1 = tf.norm(self.input-input_adv)
        loss2 = 0
        for i in range(self.output.shape[0].value):
            max_zi = tf.reduce_max(self.output[i])
            # max_zi_p = tf.argmax(output[i])

            loss2 += tf.nn.relu(max_zi - self.output[i,self.label[i]])
        loss = loss1 + self.lamda * loss2
        gra = tf.gradients(loss2, input)[0]
        input_adv = input - self.alpha * gra
        input_adv = (tf.nn.tanh(input_adv)+1)/2.

        return input_adv,self.alpha * gra

    # def diff(self,x,delta=10e-10):
    #     dx = delta
    #     dy = self.compute_loss(x + delta) - self.compute_loss(x)
    #     return dy/dx
    #
    # def attack(self, input, input_adv):
    #     gra = self.diff(input_adv)
    #     input_adv = input_adv - self.alpha * gra
    #     return input_adv, gra

def get_target_laebl(ori):
    ori = ori[0]
    ori_label = np.argmax(ori,1)
    temp = [1 for i in range(ori_label.shape[0])]
    for i in range(ori_label.shape[0]):
        if ori_label[i] == 35:
            temp[i] = -35
        elif ori_label[i] == 36:
            temp[i] = 0
    temp = np.array(temp)
    target_label = temp + ori_label
    return target_label



with tf.Session() as sess:
    captcha = CAPTCHA_creater()
    X, Y = captcha.get_batch(1)
    X = np.array(X)
    X = np.transpose(X, [0, 2, 1, 3])
    X = X[0]/255.0

    model = Training_Predict()
    model.load_model(file_path='./model/gur_ctc_model.h5base')

    input = model.base_model.input
    output = model.base_model.output


    img_place = tf.placeholder(tf.float32, shape=(100, 70, 3))
    # img_adv_place = tf.placeholder(tf.float32, shape=(100, 70, 3))
    tar_label_place = tf.placeholder(tf.int32,shape=(output.shape[1].value,))

    adversarial = Adversarial_captcha(input, output, 1, 0.01, tar_label_place)

    ori_label = sess.run(output, feed_dict={input:[X]})
    tar_label = get_target_laebl(ori_label)
    adv = adversarial.attack(img_place)

    X_adv, _= sess.run(adv,feed_dict={img_place:X,  input:[X], tar_label_place:tar_label})
    for i in range(1):
        X_adv, _ = sess.run(adv,
                                 feed_dict={img_place: X,  input: X_adv, tar_label_place: tar_label})
    adv_label = sess.run(output, feed_dict={input:X_adv})
    adv_label = np.argmax(adv_label[0],1)
    print('L2-norm metric is')
    print(np.linalg.norm(X_adv[0]-X))

    im_adv = Image.fromarray((X_adv[0]*255.).astype(np.uint8))
    im = Image.fromarray((X*255).astype(np.uint8))
    # im.show()
    im_adv.show()
    # writer = tf.summary.FileWriter("./log", sess.graph)
# writer.close()
