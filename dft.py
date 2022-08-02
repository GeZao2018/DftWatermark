import numpy as np
from matplotlib import pyplot as plt


def demo_fft():
    fig = plt.figure()
    from skimage import io
    image = io.imread('img/input.jpg')
    image = np.mean(image, axis=2)
    M, N = image.shape
    print(image.shape)
    # show image
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(image, cmap='gray')

    F = np.fft.fftn(image)
    F_magnitude = np.abs(F)
    F_magnitude = np.fft.fftshift(F_magnitude)
    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(np.log(1 + F_magnitude), cmap='gray', extent=(-N // 2, N // 2, -M // 2, M // 2))

    # set the low frequency section to 0
    K = 40
    F_shift = np.fft.fftshift(F)
    F_shift[M // 2 - K: M // 2 + K, N // 2 - K: N // 2 + K] = 0
    ax = fig.add_subplot(2, 3, 5)
    ax.imshow(np.log(1 + np.abs(F_shift)), cmap='gray', extent=(-N // 2, N // 2, -M // 2, M // 2))
    F_shift = np.fft.ifftshift(F_shift)

    image_filtered = np.real(np.fft.ifft2(F_shift))
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(image_filtered, cmap='gray')

    F_copy = F.copy()
    zero_array= np.zeros((M, N))
    K = 40
    zero_array[M // 2 - K: M // 2 + K, N // 2 - K: N // 2 + K] = 1
    F_shift = np.fft.fftshift(F_copy)
    F_shift = F_shift * zero_array
    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(np.log(1 + np.abs(F_shift)), cmap='gray', extent=(-N // 2, N // 2, -M // 2, M // 2))
    image_filtered = np.real(np.fft.ifft2(F_shift))
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(image_filtered, cmap='gray')

    plt.show()


demo_fft()