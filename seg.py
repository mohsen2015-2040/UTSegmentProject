import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from numpy.f2py.auxfuncs import throw_error


def segment(img, img_name):
    if len(img.shape) > 2:
        throw_error('invalid binary or gray image')
        return

    img = img.copy()
    img = np.array(img)
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

    h, w = img.shape[0:2]

    last_col_has_white_pixel = False
    des_img = []
    value_count = 0


    for y in range(h):
        has_white_pixel = False

        if np.any(img[y]):
            des_img.append(img[y])
            has_white_pixel = True
        if not has_white_pixel and last_col_has_white_pixel:
            value_count += 1
            res = des_img
            res = np.array(res)
            res = cv.rotate(res, cv.ROTATE_90_COUNTERCLOCKWISE)

            rows_with_white = np.any(res == 255, axis=1)
            # حذف ردیف‌هایی که سفید ندارند
            res = res[rows_with_white]
            res = cv.resize(res, (64,64), cv.INTER_CUBIC)

            path = f'./Captcha/Segmentation/'
            Path(path).mkdir(exist_ok=True)
            plt.imsave(path + f'{img_name}_v{value_count}.png', res, cmap='gray')
            des_img = []
        if not has_white_pixel:
            last_col_has_white_pixel = False
        else:
            last_col_has_white_pixel = True

    return value_count






