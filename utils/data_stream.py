
class CAPTCHA_creater:
    def __init__(self, height, length, difficult):
        pass

    def get_batch(self, batch_size):
        # return imgs, labels
        pass

    def get_one(self):
        # return img, label
        pass

    # 这俩推荐使用pickle
    def save_batch(self, path):
        pass

    def load_batch(self, path):
        # return imgs, labels
        pass

    # 这俩存图片就行
    def save_img(self, path):
        pass

    def load_img(self, path):
        # return imgs, labels
        pass
