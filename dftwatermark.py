import cv2
import numpy as np
from matplotlib import pyplot as plt

filter_m_2 = [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]

filter_m_1 = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]

filter_m_3 = [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]

filter_m = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

filter_m_16 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

def embed(img, alpha=1.35, blocksize=16, seed=147):
    '''
    img: original image with RGB 3 channels
    alpha: embedding strength
    blocksize: image block size for per bit
    seed: generating a pseudorandom sequence
    :return: watermarked image with 3 channels
    '''

    original_img = img
    original_image_g = original_img[:, :, 2]
    cover_object = original_image_g / 255
    watermark_image = np.copy(cover_object)
    Mc, Nc = np.shape(cover_object)
    max_message = Mc * Nc // (blocksize ** 2)
    message_vector = np.random.randint(2, size=max_message)
    np.random.seed(seed)
    pn_sequence_zero = np.round(2 * (np.random.random(size=np.sum(filter_m)) - 0.5))
    pn_sequence_one = np.round(2 * (np.random.random(size=np.sum(filter_m)) - 0.5))

    x, y = 0, 0
    for k in range(len(message_vector)):
        fft_block = np.fft.fft2(cover_object[x:x+blocksize, y:y+blocksize])
        abs_block = np.fft.fftshift(np.abs(fft_block))
        angle_block = np.angle(fft_block)
        # 当message_vector = 0 且 filter_m = 1 时用伪随机序列pn_sequence_zero叠加abs_block
        # 当message_vector = 1 且 filter_m = 1 时用伪随机序列pn_sequence_one 叠加abs_block
        l = 0
        if message_vector[k] == 0:
            for i in range(blocksize):
                for j in range(blocksize):
                    if filter_m[i][j] == 1:
                        abs_block_o = abs_block[i][j]
                        abs_block[i][j] = abs_block[i][j] * (1 + alpha * pn_sequence_zero[l])
                        abs_block[blocksize-i-1][blocksize-j-1] = abs_block[blocksize-i-1][blocksize-j-1] + abs_block[i][j] - abs_block_o
                        l += 1
        else:
            for i in range(blocksize):
                for j in range(blocksize):
                    if filter_m[i][j] == 1:
                        abs_block_o = abs_block[i][j]
                        abs_block[i][j] = abs_block[i][j] * (1 + alpha * pn_sequence_one[l])
                        abs_block[blocksize-i-1][blocksize-j-1] = abs_block[blocksize-i-1][blocksize-j-1] + abs_block[i][j] - abs_block_o
                        l += 1

        abs_block = np.fft.ifftshift(abs_block)
        watermark_image[x:x+blocksize, y:y+blocksize] = np.abs(np.fft.ifft2(abs_block * np.exp(complex(0, -1) * angle_block)))
        if x + blocksize >= Mc:
            x, y = 0, y + blocksize
        else:
            x += blocksize

    watermark_image_g = watermark_image * 255
    plt.subplot(121), plt.imshow(original_img), plt.title('Cover'), plt.axis('off')
    original_img[:, :, 2] = watermark_image_g
    plt.subplot(122), plt.imshow(original_img), plt.title('Marked'), plt.axis('off')
    plt.show()
    return original_img, message_vector


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y


def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = np.dot(a, b).sum() / np.sqrt(np.dot(a, a).sum() * np.dot(b, b).sum())
    return r


def recover(img, blocksize=16, seed=147):
    '''
    :param img: watermarked image with RGB 3 channels
    :param blocksize: image block size for per bit
    :param seed: generating a pseudorandom sequence
    :return: message vector of watermark information
    '''

    watermark = img[:, :, 2]
    watermark_img = watermark / 255
    Mc, Nc = np.shape(watermark_img)
    max_message = Mc * Nc // (blocksize ** 2)
    message_vector = np.empty(max_message, dtype=int)
    np.random.seed(seed)
    pn_sequence_zero = np.round(2 * (np.random.random(size=np.sum(filter_m)) - 0.5))
    pn_sequence_one = np.round(2 * (np.random.random(size=np.sum(filter_m)) - 0.5))
    correlation_zero = []
    correlation_one = []

    x, y = 0, 0
    for k in range(len(message_vector)):
        fft_block = np.fft.fft2(watermark_img[x:x+blocksize, y:y+blocksize])
        abs_block = np.fft.fftshift(np.abs(fft_block))
        sequence = []
        for i in range(blocksize):
            for j in range(blocksize):
                if filter_m[i][j] == 1:
                    sequence.append(abs_block[i][j])

        correlation_zero.append(corr2(pn_sequence_zero, np.array(sequence, dtype=np.float64)))
        correlation_one.append(corr2(pn_sequence_one, np.array(sequence, dtype=np.float64)))
        if x + blocksize >= Mc:
            x, y = 0, y + blocksize
        else:
            x += blocksize

    for k in range(len(message_vector)):
        if correlation_one[k] > correlation_zero[k]:
            message_vector[k] = 1
        else:
            message_vector[k] = 0

    return message_vector


if __name__ == '__main__':
    # print(np.sum(filter_m))
    # print(np.sum(filter_m_16))
    img = cv2.imread('img/input.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_marked, message_embed = embed(img)
    # img_marked = cv2.GaussianBlur(img_marked, (5, 5), 5)
    # img_marked = cv2.blur(img_marked, (5, 5))
    # img_marked = cv2.medianBlur(img_marked, 5)
    # img_marked = cv2.resize(img_marked, (600, 600))
    message_recover = recover(img_marked)
    print(len(message_embed))
    cnt, total = 0, len(message_embed)
    for i in range(total):
        if message_recover[i] == message_embed[i]:
            cnt += 1
    print('accuracy:', cnt / total)