# encoding:utf8
import numpy as np
import freetype as ft
import cv2

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 256
MAX_CAPTCHA = 18
CHAR_SET_LEN = 10


class TextGenerator(object):
    def __init__(self, ttf):
        self.face = ft.Face(ttf)     # ttf是字体名称字符串'fonts/OCR-B.ttf'

    def draw_text(self, image, pos, text, text_size, color):
        self.face.set_char_size(text_size * 64)

        pen = ft.Vector()
        pen.x = pos[0]
        pen.y = pos[1] + 10   #

        img = np.copy(image)  # image只是个空的模板，每次返回的是image的含有内容的副本
        for cur_char in text:
            self.face.load_char(cur_char)
            bitmap = self.face.glyph.bitmap

            for row in range(bitmap.rows):          # 画布矩阵img和bitmap.buffer中的图像都是位于第四象限的图像
                for col in range(bitmap.width):     # 即上边界为x轴，左边界为y轴
                    if bitmap.buffer[row * bitmap.width + col] != 0:
                        img[pen.y + row][pen.x + col][0] = color[0]
                        img[pen.y + row][pen.x + col][1] = color[1]
                        img[pen.y + row][pen.x + col][2] = color[2]

            pen.x += self.face.glyph.advance.x >> 6  # 转化为像素值，每复制完一个字符的像素便向前推进一段距离
            # pen.x += 4 # 手动调节字符间距
            print(pen.x)
        return img


class GenIdCard(object):
    def __init__(self):
        self.textGenerator = TextGenerator("endToEndByCnn\\OCR-B 10 BT.ttf")
        self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.char_set = self.number
        self.len = len(self.char_set)
        self.max_size = 18      # 生成18个数字

    # 随机生成字串，长度固定，返回text,及对应的向量
    def random_text(self):
        text = ''
        vecs = np.zeros((self.max_size * self.len))  # 长度为18*10的一维数组
        for i in range(self.max_size):
            c = np.random.choice(self.char_set)
            vec = self.char2vec(c)   # 0 1 0 0 0 0 0 0 0 0
            text = text + c
            vecs[i * self.len:(i + 1) * self.len] = np.copy(vec)  # 将第i个随机数字的向量vec添加到vecs中的对应位置处
        return text, vecs

    # 比如一个结果为415195757106768950
    # [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
    # 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
    # 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.
    # 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0.
    # 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.
    # 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.
    # 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0.
    # 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

    # 单字转向量
    def char2vec(self, c):
        vec = np.zeros([self.len])  # 长度为10的一维数组
        for j in range(self.len):
            if self.char_set[j] == c:
                vec[j] = 1
        return vec

    # 根据生成的text，生成image,返回标签和图片元素数据
    def gen_image(self, flag=False):
        if flag:  # flag为True表示生成随机长度的数字串，为False表示生成长度为18的数字串
            self.max_size = np.random.randint(1, 19)  # max_size设置为区间[1,19)内的随机整数
        text, vec = self.random_text()
        image = np.zeros([32, 256, 3])      # 生成高32*宽256的3通道图像矩阵画布
        color = (255, 255, 255)             # Write
        position = (0, 0)                   # 指定文字书写位置
        text_size = 13                      # 指定字体大小
        image = self.textGenerator.draw_text(image, position, text, text_size, color)
        return image[:, :, 0], text, vec    # 仅返回一个通道的值即可，颜色对于汉字识别没有什么意义

    # 向量转文本
    def vec2text(self, vecs):
        text = ''
        v_len = len(vecs)
        for i in range(v_len):
            if vecs[i] == 1:
                text = text + self.char_set[i % self.len]
        return text

if __name__ == '__main__':
    import pdb
   #  pdb.set_trace()
    genObj = GenIdCard()
    image_data, label, vec = genObj.gen_image()
    # print(label, vec)

    cv2.imshow('title', image_data)  # 这里的image_data既可以是不含通道的二维图像，也可以是含有通道的三维图像，
    cv2.waitKey(0)



    # img = np.zeros([300, 300, 3])
    # obj1 = PutChineseText("d:\\tf\\OCR-B 10 BT.ttf")
    # obj2 = PutChineseText("d:\\tf\\fzskbxkt.ttf")   # 这里无法打开中文命名的文件：方正宋刻本秀楷简体.ttf，所以只能重命名一下
    # image1 = obj1.draw_text(img, (50, 50), '1232142153253215', 20, (255, 255, 255))
    # image2 = obj2.draw_text(img, (3, 3), '湖南省邵阳县', 20, (255, 255, 255))
    #
    # cv2.imshow('ss', image1)
    # cv2.imshow('image1', image2)
    # cv2.waitKey(0)
