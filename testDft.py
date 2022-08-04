import cv2
import math
import numpy as np
# from matplotlib import pyplot as plt

points = []


def get_points(filename, dis=100):
    global points

    h, w = 400, 400
    for i in range(h):
        for j in range(201):
            if int(math.sqrt(pow(i-200, 2) + pow(j-200, 2))) == dis:
                points.append((j, i))

    img = cv2.imread(filename)
    for r, c, in points:
        img[r][c] = 0
    cv2.imshow('img', img)
    cv2.waitKey()


def rotate_image(image, angle):
    '''
     . 旋转图片
     . @param image    opencv读取后的图像
     . @param angle    (逆)旋转角度
    '''

    h, w = image.shape[:2]  # 返回(高,宽,色彩通道数),此处取前两个值返回
    newW = int(h * np.fabs(np.sin(np.radians(angle))) + w * np.fabs(np.cos(np.radians(angle))))
    newH = int(w * np.fabs(np.sin(np.radians(angle))) + h * np.fabs(np.cos(np.radians(angle))))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    M[0, 2] += (newW - w) / 2
    M[1, 2] += (newH - h) / 2
    return cv2.warpAffine(image, M, (newW, newH), borderValue=(255, 255, 255))


def embed(filename):
    original_image = cv2.imread(filename)
    original_image_b = original_image[:, :, 0]
    watermark_image = original_image_b / 255

    fft_image = np.fft.fft2(watermark_image)
    abs_image = np.fft.fftshift(np.abs(fft_image))
    angle_image = np.angle(fft_image)

    watermark = ''
    for i in range(1, len(points)):
        if abs_image[points[i-1][0]][points[i-1][1]] < abs_image[points[i][0]][points[i][1]]:
            watermark += '1'
        else:
            watermark += '0'

    return watermark


def test_embed(filename):
    original_image = cv2.imread(filename)

    # resized_image = cv2.resize(original_image, (350, 350))
    # cv2.imwrite('img/temp.jpg', resized_image)
    rotated_image = rotate_image(original_image, 180)
    cv2.imwrite('img/temp.jpg', rotated_image)
    original_image = cv2.imread('img/temp.jpg')
    original_image = cv2.resize(original_image, (400, 400))

    original_image_b = original_image[:, :, 0]

    # original_image_b = cv2.blur(original_image_b, (7, 7))
    # original_image_b = cv2.GaussianBlur(original_image_b, (7, 7), 250)
    # original_image_b = cv2.medianBlur(original_image_b, 5)
    # original_image_b[0:20, :] = original_image_b[380:400, :] = 0
    # original_image_b[:, 0:20] = original_image_b[:, 380:400] = 0

    cv2.imshow('original', original_image_b)
    cv2.waitKey()

    watermark_image = original_image_b / 255

    fft_image = np.fft.fft2(watermark_image)
    abs_image = np.fft.fftshift(np.abs(fft_image))
    angle_image = np.angle(fft_image)

    watermark = ''
    for i in range(1, len(points)):
        if abs_image[points[i - 1][0]][points[i - 1][1]] < abs_image[points[i][0]][points[i][1]]:
            watermark += '1'
        else:
            watermark += '0'

    return watermark


def check_watermark(mark1, mark2):
    cnt = 0
    for i in range(len(mark1)):
        if mark1[i] == mark2[i]:
            cnt += 1
    print(cnt, len(mark1), cnt / len(mark1))


if __name__ == '__main__':
    filename = './img/input.jpg'

    get_points(filename, dis=32)

    check_watermark(embed(filename), test_embed(filename))

    # embed(filename)
    # test_embed(filename)