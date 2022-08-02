import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
import random


def dwt_embed(img_gray, img_watermark, seed=2020):
    "An illustration of how data are embedded in pair-wise DCT coefficients,"
    " img_gray - of grayscale"
    " img_watermark - the to be embedded msg composed of 0 and 1 only"
    " seed - the encryption password"

    if len(img_gray.shape) > 2 or len(img_watermark.shape) > 2:
        print("Parameter img should be of grayscale")
        return img_gray

    # Step 1: DWT in level 2 Haar coefficients cH_l2 and cV_l2
    cA_l1, (cH_l1, cV_l1, cD_l1) = pywt.dwt2(img_gray.astype(np.float32), 'haar')
    cA_l2, (cH_l2, cV_l2, cD_l2) = pywt.dwt2(cA_l1, 'haar')

    # Step 2: Embed
    height, width = img_gray.shape
    img_watermark = cv2.resize(img_watermark, (width >> 2, height >> 2))
    img_watermark = img_watermark.astype(np.float32)

    # change 0 to -1
    # img_watermark[img_watermark<1] = -1
    alpha = 3  # The strength of watermark
    cH_l2 = alpha * img_watermark
    cV_l2 = alpha * img_watermark

    # Step 3: IDWT
    cA_l1 = pywt.idwt2((cA_l2, (cH_l2, cV_l2, cD_l2)), 'haar')
    img_marked = pywt.idwt2((cA_l1, (cH_l1, cV_l1, cD_l1)), 'haar')

    return img_marked.astype(np.uint8)


# Non-blind detection, requires the original watermark
def dwt_extract(img_marked, img_watermark, seed=2020):
    "An illustration of data extraction to the previous embedding,"
    " img_marked - of grayscale"
    " img_watermark - Non-blind detection, requires the original watermark"
    " seed - the password for decryption"

    if len(img_marked.shape) > 2:
        print("Parameter img should be of grayscale")
        return img_marked

    # Step 1: DWT in level 2 Haar coefficients cH_l2 and cV_l2
    cA_l1, (cH_l1, cV_l1, cD_l1) = pywt.dwt2(img_marked.astype(np.float32), 'haar')
    cA_l2, (cH_l2, cV_l2, cD_l2) = pywt.dwt2(cA_l1, 'haar')

    # Step 2: Extract
    height, width = img_marked.shape
    img_watermark = cv2.resize(img_watermark, (width >> 2, height >> 2))
    img_watermark = img_watermark.astype(np.float32)
    # img_watermark[img_watermark<1] = -1
    alpha = 3
    img_watermark_extracted = cH_l2 * img_watermark + cV_l2 * img_watermark
    img_watermark_extracted = 255 * img_watermark_extracted / np.max(img_watermark_extracted)
    img_watermark_extracted[img_watermark_extracted < alpha] = 0
    img_watermark_extracted[img_watermark_extracted >= alpha] = 255
    return img_watermark_extracted.astype(np.uint8)


if __name__ == '__main__':
    img_gray = cv2.imread('img/cover.png', cv2.IMREAD_GRAYSCALE)

    img_watermark = cv2.imread('img/mark.png', cv2.IMREAD_GRAYSCALE)
    _, img_watermark = cv2.threshold(img_watermark, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_marked = dwt_embed(img_gray, img_watermark, 20200417)

    cv2.imwrite('img/cover_marked.png', img_marked)

    # print(img_marked.shape, type(img_marked), type(img_marked[0,0]))
    img_stego = cv2.imread('img/cover_marked.png', cv2.IMREAD_GRAYSCALE)
    img_watermark = cv2.imread('img/mark.png', cv2.IMREAD_GRAYSCALE)
    _, img_watermark = cv2.threshold(img_watermark, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_watermark_extracted = dwt_extract(img_stego, img_watermark, 20200417)

    plt.figure(figsize=(4, 3))
    plt.subplot(221), plt.imshow(img_gray, cmap='gray'), plt.title('Cover'), plt.axis('off')
    plt.subplot(222), plt.imshow(img_marked, cmap='gray'), plt.title('Marked'), plt.axis('off')
    plt.subplot(223), plt.imshow(img_watermark, cmap='gray'), plt.title('Watermark'), plt.axis('off')
    plt.subplot(224), plt.imshow(img_watermark_extracted, cmap='hot'), plt.title('Watermark Extracted'), plt.axis('off')
    plt.tight_layout()
    plt.show()
