import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt




def get_laplacian_filter(size):
    arr = []
    for j in range(size):
        arr.append([])
        for i in range(size):
            if j != int(size / 2) and i == int(size / 2):
                arr[j].append(-1)
            elif j == i and i == int(size / 2):
                arr[j].append(2*size - 2)
            elif j == int(size / 2) and i != int(size / 2):
                arr[j].append(-1)
            else:
                arr[j].append(0)

    return arr

def get_high_boost_filter(size, a):
    arr = []
    for j in range(size):
        arr.append([])
        for i in range(size):
            if i == j and i == int(size / 2):
                arr[j].append(a - 1/size**2)
            else:
                arr[j].append(-1/size**2)

    return arr

def apply_sharpening_filter(img, filter_matrix, stream, cmap = None):
    img = img.copy()
    h, w = img.shape[:2]
    filter_matrix = np.array(filter_matrix)
    fh, fw = filter_matrix.shape

    # create output image
    out = np.array(img)

    for j in range(h - fh + 1):
        for i in range(w - fw + 1):
            acc = 0  # for RGB channels

            for fj in range(fh):
                for fi in range(fw):
                    acc += img[j + fj, i + fi] * filter_matrix[fj, fi]

            # clamp values
            acc = np.clip(acc, 0, 255)

            center_j = j + fh // 2
            center_i = i + fw // 2
            out[center_j, center_i] = acc.astype(np.uint8)

    if cmap is not None:
        plt.imsave(stream, out, cmap=cmap)
    else:
        plt.imsave(stream, out)

    return out


