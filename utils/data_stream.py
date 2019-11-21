import random
import string
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import pickle
import glob
import tensorflow as tf

class CAPTCHA_creater:
    def __init__(self, height, length, difficult):
        self.font_path = 'Arial.ttf'
        # 生成几位数的验证码
        self.number = 4
        # 生成验证码图片的高度和宽度
        self.height = height
        self.length = length
        self.difficult = difficult

    # 生成样本，我觉得我写了个大bug，希望你可以心平气和地看完这段代码。。。

    def get_batch(self, batch_size):

        # 产生图片
        for j_train in range(batch_size):

            # 这里地j_train是指图片的序号，在保存图片时用得到

            self.get_one(j_train)

        # 加载图片

        image, label= self.load_img("")

        # load_img返回的image是所有image的路径，所以在这里要给它们转码

        imgs = tf.image.decode_png(image, channels=3)


        # 生成batch

        image_batch, label_batch = tf.train.batch([imgs, label],
                                                  batch_size=batch_size,
                                                  num_threads=32)

        # 重新排列label，行数为[batch_size]

        self.label_batch = tf.reshape(label_batch, [batch_size])
        self.image_batch = tf.cast(image_batch, tf.float32)


        # return imgs, labels

        return self.image_batch,self.label_batch


#生成一个图像
    def get_one(self,j):

       #这里默认difficult的值为0/1/2，分别代表easy middle hard

       #在不同难度下，参数的设置

        if self.difficult == 0:
            draw_line = False
            draw_circle = False
            rotate = False
            bgcolor = (255, 255, 255)
        if self.difficult == 1:
            draw_line = True
            draw_circle = False
            rotate = True
            point_number = random.randint(5, 10)
            line_number = random.randint(3, 5)
            bgcolor = (232, 232, 232)
        if self.difficult == 2:
            draw_line = True
            draw_circle = True
            rotate = True
            point_number = random.randint(10, 20)
            arc_number = random.randint(5, 10)
            line_number = random.randint(10, 15)
            bgcolor = (136, 136, 136)


        #底下全是生成验证码的操作，应该没问题

        image = Image.new('RGBA', (self.width, self.height), bgcolor)  # 创建图片
        font = ImageFont.truetype(self.font_path, 25)  # 验证码的字体
        draw = ImageDraw.Draw(image)  # 创建画笔
        text = self.gene_text()  # 生成字符串
        self.label.append(text)
        font_width, font_height = font.getsize(text)
        draw.text(((self.width - font_width) / self.number, (self.height - font_height) / self.number), text,
                  font=font, fill=self.getRandomColor1())  # 填充字符串

        if draw_line:
            self.gene_line(draw, self.width, self.height, line_number)
            self.drawPoint(draw, self.width, self.height, point_number)
        if draw_circle:
            self.drawArc(draw, self.width, self.height, arc_number)

        if rotate:
            image = image.rotate(random.randint(-10, 20))
            draw = ImageDraw.Draw(image)  # 创建画笔

            for x in range(self.width):
                for y in range(self.height):
                    c = image.getpixel((x, y))
                    if c == (0, 0, 0, 0):
                        draw.point([x, y], fill=bgcolor)


        #每生成一个图片，我就保存一个

        self.save_img("", image, text, j)


        return image, text
        # return img, label



    # 这俩推荐使用pickle

    def save_batch(self, path):


        # 我把path规定死了

        file_img_path = self.img_dir + "/hahardness" + str(self.difficult) + '_imgs.pkl'
        file_label_path = self.img_dir+"/hahardness" + str(self.difficult) + '_labels.pkl'

        file_img= open(file_img_path,'wb')
        file_label = open(file_label_path,'wb')

        pickle.dump(self.image_batch, file_img)
        pickle.dump(self.label_batch,file_label)

        file_label.close()
        file_img.close()

    def load_batch(self, path):
        try:
            file_img_path = self.img_dir + "/hahardness" + str(self.difficult) + '_imgs.pkl'
            file_label_path = self.img_dir + "/hahardness" + str(self.difficult) + '_labels.pkl'

            file_img = open(file_img_path, 'rb')
            file_label = open(file_label_path, 'rb')
            imgs = pickle.load(file_img)
            labels = pickle.load(file_label)
            file_label.close()
            file_img.close()

            return imgs, labels
            # return imgs, labels
        except:
            print("you haven't saved the batch")


    # 这俩存图片就行
    def save_img(self, path,image,text,j):
        self.self.img_dir = self.build_file_path('train/hardness' + str(self.difficult))
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        image.save(self.img_dir + '/hardness' + str(self.difficult) + '_' + str(j + 1) + '.png')  # 保存验证码图片

        # 我把验证码的字母存在一个txt里，我也不知道有没有用

        self.path_file_name = self.img_dir + '/hardness' + str(self.difficult) + '.txt'
        if not os.path.exists(self.path_file_name):
            with open(self.path_file_name, "a") as f:
                print(f)
        with open(self.path_file_name, "a") as f:
            f.write(text + '\n')


    # 我不太确定这里的img是不是这样子读的

    def load_img(self, path):
        imgs_paths = glob.glob(os.path.join(self.img_dir+"/", '*.png'))
        image_contents = tf.read_file(imgs_paths)
        return image_contents,self.labels
        # 这里返回的是image的所有路径，和labels。labels我把它当成class里的一个变量了，所以用的self





    # the end

    # 剩下的都是并不是很重要的函数

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

        # 用来随机生成一个字符串

    def gene_text(self):
        source = list(string.ascii_uppercase)
        for index in range(0, 10):
            source.append(str(index))
        return ''.join(random.sample(source, 4))  # number是生成验证码的位数

        # 用来绘制干扰线

    def gene_line(self,draw, width, height, line_number):
        for line in range(line_number):
            begin = (random.randint(0, width), random.randint(0, height))
            end = (random.randint(0, width), random.randint(0, height))
            linecolor = (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))
            draw.line([begin, end], fill= self.getRandomColor2())

        # 用来绘制干扰点

    def drawPoint(self,draw, width, height, point_number):
        for i in range(point_number):
            x = random.randint(0, width)
            y = random.randint(0, height)
            draw.point((x, y), fill= self.getRandomColor2())

        # 用来绘制干扰圆弧

    def drawArc(self,draw, width, height, arc_number):
        for i in range(arc_number):
            x = random.randint(0, width)
            y = random.randint(0, height)
            draw.arc((x, y, x + 8, y + 8), 0, 90, fill= self.getRandomColor2())

        # 添加文件路径

    def build_file_path(self,x):
        if not os.path.isdir('./imgs'):
            os.mkdir('./imgs')
        return os.path.join('./imgs', x)

