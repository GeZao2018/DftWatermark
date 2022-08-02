import cv2
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class DftWatermark:
    def __init__(self, filename):
        self.img = cv2.imread(filename)
        self.img_f = np.fft.fft2(self.img)
        self.img_sf = np.fft.fftshift(self.img_f)
        self.mark = np.random.randint(2, size=16)
        self.mark[self.mark == 0] = -1
        self.alpha = 500.0
        np.random.seed(23)
        self.s = []
        for i in range(45):
            t = np.random.randint(2, size=16)
            t[t == 0] = -1
            self.s.append(t)


    def encode(self):
        h, w, c = self.img.shape
        half_h, half_w = h // 2, w // 2
        idx = 0
        x = []
        for i in range(half_h-5, half_h+5):
            for j in range(half_w-5, half_w+5):
                if i >= j: continue
                sum = 0
                for k in range(16):
                    sum += self.s[idx][k] * self.mark[k]
                self.img_sf[i][j] += self.alpha * sum
                self.img_sf[j][i] += self.alpha * sum
                x.append(self.img_sf[i][j][0])
                idx += 1

        print(x)
        img_if = np.fft.ifftshift(self.img_sf)
        img_i = np.abs(np.fft.ifft2(img_if))
        cv2.imwrite('img/input_mark.jpg', img_i)

    def decode(self):
        img = cv2.imread('img/input_mark.jpg')
        img_f = np.fft.fft2(img)
        img_sf = np.fft.fftshift(img_f)
        h, w, c = img.shape
        half_h, half_w = h // 2, w // 2
        mark = []
        x = []
        for i in range(half_h - 5, half_h + 5):
            for j in range(half_w - 5, half_w + 5):
                if i >= j: continue
                x.append(img_sf[i][j][0])

        mean = np.mean(np.real(x))
        for i in range(16):
            p = 0.
            for j in range(45):
                p += (x[j] - mean) * self.s[j][i]
            if p > 0:
                mark.append(1)
            else:
                mark.append(0)

        print(x)
        self.mark[self.mark == -1] = 0
        print(self.mark.tolist())
        print(mark)


def func2(filename, filter_size=9):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_u, img_v = img[:, :, 1], img[:, :, 2]

    h, w, c = img.shape
    res = np.full((h, w), 255)
    step = filter_size // 2
    cnt = 0
    for i in range(step, h - step):
        for j in range(step, w - step):
            block_u = img_u[i-step:i+step+1, j-step:j+step+1]
            block_v = img_v[i-step:i+step+1, j-step:j+step+1]
            avg1, avg2 = np.mean(block_u), np.mean(block_v)
            if img_u[i][j] >= avg1+20 or img_v[i][j] >= avg2+20:
                res[i][j] = 0
                cnt += 1

    print(cnt)
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show()
    cv2.imwrite("img/output.jpg", res)
    return res


if __name__ == '__main__':
    filename = 'img/input.jpg'
    # func2(filename)
    dft = DftWatermark(filename)
    dft.encode()
    dft.decode()