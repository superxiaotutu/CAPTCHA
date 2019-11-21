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

image_channel = 3
image_height = 64
image_width = 192

def binary(image):
    image = image.convert('L')
    image = image.point(lambda x: 255 if x > np.mean(image) else 0)
    image = image.convert('RGB')
    return image

def add_gauss(image, radius=2):
    image = cv2.GaussianBlur(image,(radius,radius))
    return image

def sigmoid(image, w, t):

    img = np.array(image)
    img = (img[:,:,0] * 299 + img[:,:,1] * 587 + img[:,:,2] * 114 + 500) / 1000
    return img

