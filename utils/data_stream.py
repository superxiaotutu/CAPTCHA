import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import pickle
from utils.config import *
import matplotlib.pyplot as plt
import numpy as np


class CAPTCHA_creater:
    def __init__(self):
        self.font_path = 'utils/Arial.ttf'
        # 生成几位数的验证码
        self.number = NumCAPTCHA
        # 生成验证码图片的高度和宽度
        self.height, self.width = image_height, image_width
        self.difficult = diff
        self.source = list(string.ascii_uppercase)
        for index in range(0, 10):
            self.source.append(str(index))

        if self.difficult == 0:
            self.draw_line = False
            self.draw_circle = False
            self.rotate = False
            self.bgcolor = (255, 255, 255)
        elif self.difficult == 1:
            self.draw_line = True
            self.draw_circle = False
            self.rotate = True
            self.point_number = random.randint(5, 10)
            self.line_number = random.randint(3, 5)
            self.bgcolor = (232, 232, 232)
        else:
            self.draw_line = True
            self.draw_circle = True
            self.rotate = True
            self.point_number = random.randint(10, 20)
            self.arc_number = random.randint(5, 10)
            self.line_number = random.randint(10, 15)
            self.bgcolor = (136, 136, 136)

    # 生成样本，我觉得我写了个大bug，希望你可以心平气和地看完这段代码。。。
    # 上面的注释留作纪念， 我的代码一般不留注释的。。。
    def get_batch(self, batch_size):
        img_lst = []
        label_lst = []
        for i in range(batch_size):
            tmp_img, tmp_lable = self.get_one()
            tmp_lable = self.text2onehot(tmp_lable)
            img_lst.append(np.asarray(tmp_img))
            label_lst.append(tmp_lable)
        return [img_lst, np.asarray(label_lst)]

    def get_one(self):
        image = Image.new('RGB', (self.height, self.width), self.bgcolor)  # 创建图片
        font = ImageFont.truetype(self.font_path, 25)  # 验证码的字体
        draw = ImageDraw.Draw(image)  # 创建画笔
        text = self.gene_text()  # 生成字符串
        font_width, font_height = font.getsize(text)
        draw.text(((self.width - font_width) / self.number, (self.height - font_height) / self.number), text,
                  font=font, fill=self.getRandomColor1())  # 填充字符串
        if self.draw_line:
            self.gene_line(draw, self.width, self.height, self.line_number)
            self.drawPoint(draw, self.width, self.height, self.point_number)
        if self.draw_circle:
            self.drawArc(draw, self.width, self.height, self.arc_number)
        if self.rotate:
            image = image.rotate(random.randint(-10, 20))
            draw = ImageDraw.Draw(image)  # 创建画笔
            for x in range(self.width):
                for y in range(self.height):
                    c = image.getpixel((x, y))
                    if c == (0, 0, 0, 0):
                        draw.point([x, y], fill=self.bgcolor)
        return image, text

    def onehot2text(self, oh):
        text = ''
        for i in np.argmax(oh, axis=1):
            text += self.source[i]
        return text


    def text2onehot(self, text):
        oh = np.zeros([NumCAPTCHA, NumAlb])
        for i in range(NumCAPTCHA):
            oh[i][self.source.index(text[i])] = 1
        return oh


    def save_batch(self, path, batch=None, batchsize=50):
        if not batch:
            batch = self.get_batch(batchsize)
        file_batch = open(path, 'wb')
        pickle.dump(batch, file_batch)
        file_batch.close()

    def load_batch(self, path):
        try:
            file_batch = open(path, 'rb')
            batch = pickle.load(file_batch)
            file_batch.close()
            return batch
        except:
            print("you haven't saved the batch")

    def save_img(self, path, image=None, text=None):
        if not (image or text):
            image, text = self.get_one()
        path = path + "/" + str(text) + ".png"
        plt.imsave(path, image)

    def load_img(self, path):
        imgs = plt.imread(path)
        labels = path.split('/')[-1].split('.')[0]
        return imgs, labels

    def getRandomColor1(self):
        r = random.randint(32, 127)
        g = random.randint(32, 127)
        b = random.randint(32, 127)
        return (r, g, b)

    def getRandomColor2(self):
        r = random.randint(64, 255)
        g = random.randint(64, 255)
        b = random.randint(64, 255)
        return (r, g, b)

    def gene_text(self):
        return ''.join(random.sample(self.source, 4))

    def gene_line(self, draw, width, height, line_number):
        for line in range(line_number):
            begin = (random.randint(0, width), random.randint(0, height))
            end = (random.randint(0, width), random.randint(0, height))
            linecolor = (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))
            draw.line([begin, end], fill=self.getRandomColor2())

    def drawPoint(self, draw, width, height, point_number):
        for i in range(point_number):
            x = random.randint(0, width)
            y = random.randint(0, height)
            draw.point((x, y), fill=self.getRandomColor2())

    def drawArc(self, draw, width, height, arc_number):
        for i in range(arc_number):
            x = random.randint(0, width)
            y = random.randint(0, height)
            draw.arc((x, y, x + 8, y + 8), 0, 90, fill=self.getRandomColor2())

# if __name__ == '__main__':
    # test = CAPTCHA_creater()
    # im, la = test.get_one()
    # plt.imshow(im)
    # plt.show()
    # print(la)
    # tutu = test.text2onehot('DFGD')
    # print(tutu)
    # tutu = test.onehot2text(tutu)
    # print(tutu)