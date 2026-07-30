import numpy as np
import cv2 as cv

def cosine_similarity(img1, img2):
    v1 = img1[1].astype(np.float32).flatten()
    v2 = img2[1].astype(np.float32).flatten()
    v1 = np.clip(v1, 1, 255)# .where(v1 == 0, 1, v1)
    v2 = np.clip(v2, 1, 255)# .where(v2 == 0, 1, v2)
    dot = np.dot(v1, v2)  # now float32
    norm_1 = np.linalg.norm(v1)  # float32
    norm_2 = np.linalg.norm(v2)
    return dot / (norm_1 * norm_2)

def similarity(img1, img2):
    img1 = img1.copy()
    img2 = img2.copy()
    img1 = np.array(img1)
    img2 = np.array(img2)

    img1 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY)
    img2 = cv.threshold(img2, 0, 255, cv.THRESH_BINARY)

    sim_rate = cosine_similarity(img1, img2)
    return sim_rate