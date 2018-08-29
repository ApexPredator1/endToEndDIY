# encoding:utf8
import numpy as np
import freetype as ft
import cv2

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 256
MAX_CAPTCHA = 18
CHAR_SET_LEN = 10

# 该文件可以用来生成任意长度的汉字、数字、字母等

class PutChineseText(object):
    def __init__(self, ttf):
        self._face = ft.Face(ttf)     # ttf是字体名称字符串'fonts/OCR-B.ttf'

    def draw_text(self, image, pos, text, text_size, text_color):
        """
            draw chinese(or not) text with ttf
            :param image:     image(numpy.ndarray) to draw text
            :param pos:       where to draw text
            :param text:      the context, for chinese should be unicode type
            :param text_size: text size
            :param text_color:text color
            :return:          image
        """
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender / 64.0
        ypos = int(ascender)
        # descender = metrics.descender/64.0
        # height = metrics.height/64.0
        # linegap = height - ascender + descender

        x_pos = pos[0] + 2               # pos[0]和pos[1]均为0，
        y_pos = pos[1] + 8 #  + ypos       # 加上上行高度，


        pen = ft.Vector()
        pen.x = x_pos << 6  # 像素长度x_pos乘以64转化为26.6格式的值
        pen.y = y_pos << 6

        # hscale = 1.0
        # matrix = ft.Matrix(int(hscale) * 0x10000, int(0.2 * 0x10000), int(0.0 * 0x10000), int(1.1 * 0x10000))
        # prev_char = 0
        # cur_pen = ft.Vector()
        # pen_translate = ft.Vector()

        image = np.copy(image)
        for cur_char in text:
            # self._face.set_transform(matrix, pen_translate)
            self._face.load_char(cur_char)
            # kerning = self._face.get_kerning(prev_char, cur_char)
            # pen.x += kerning.x
            # cur_pen.x = pen.x
            # cur_pen.y = pen.y\
                        # - self._face.glyph.bitmap_top * 64
            # self.draw_ft_bitmap(image, self._face.glyph.bitmap, cur_pen, text_color)
            self.draw_ft_bitmap(image, self._face.glyph.bitmap, pen, text_color)
            pen.x += self._face.glyph.advance.x
            # prev_char = cur_char

        return image

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        """
        draw each char
        :param img:    image
        :param bitmap: bitmap
        :param pen:    pen
        :param color:  pen color e.g.(0,0,255) - red
        :return:       image
        """
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows
        glyph_pixels = bitmap.buffer

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row * cols + col] != 0:
                    img[y_pos + row][x_pos + col][0] = color[0]
                    img[y_pos + row][x_pos + col][1] = color[1]
                    img[y_pos + row][x_pos + col][2] = color[2]


class GenIdCard(object):
    def __init__(self):
        self.ft = PutChineseText("OCR-B 10 BT.ttf")
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

    # 比如一个结果为text是数字字符串：415195757106768950，vecs是列表：
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
        if flag:                            # flag为True表示生成随机长度的数字串，为False表示生成长度为18的数字串
            self.max_size = np.random.randint(1, 19)  # max_size设置为区间[1,19)内的随机整数
        text, vec = self.random_text()
        image = np.zeros([32, 256, 3])      # 生成高32*宽256的3通道图像矩阵画布
        color = (255, 255, 255)             # Write
        position = (0, 0)                   # 指定文字书写位置
        text_size = 23                      # 指定字体大小
        image = self.ft.draw_text(image, position, text, text_size, color)
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
    image_data, label, vec = genObj.gen_image(True)
    print(label, vec)

    cv2.imshow('title', image_data)
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
