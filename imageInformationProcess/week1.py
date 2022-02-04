import numpy as np
from matplotlib import pyplot as plt

# file = open('C:\\Users\WangYao\Documents\Course note\ImageInformationProcess\TA\Week 1-20210615file.txt', 'w')
# file.write('hello word \n')
# file.write('this is our new text file \n')
# file.write('this is our new text file111111 \n')

import cv2
img = cv2.imread('C:\\Users\WangYao\Documents\Course note\ImageInformationProcess\TA\Week1\orange.jpg')
print(img)

_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(_img)
plt.show()

resize_image = cv2.resize(_img, (500, 500))
plt.imshow(resize_image)
plt.show()

resize_image = cv2.resize(_img, (0, 0), fx=1/2, fy=1/2)
plt.imshow(resize_image)
plt.show()

# 截图
x1, y1 = 100, 200
x2, y2 = 150, 300

cropped_image = _img[y1 : y2, x1: x2, :]
plt.imshow(cropped_image)
plt.show()

# 将图片转化为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()


