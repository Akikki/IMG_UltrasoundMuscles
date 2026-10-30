import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
from pathlib import Path
from skimage.measure import ransac, LineModelND
import os
from resize import main_region, crop_image
from aponevroses import detect_ap
import skimage as ski

# Nimg = 0000, 0001, 0002 jusquà' 1047

def verif(Nimg):
    img = cv.imread("data/apo_imgs_v1/apo_images_new_model_v1/image_"+Nimg+".tif", cv.IMREAD_GRAYSCALE)
    mask = cv.imread("data/apo_masks_v1/apo_masks_new_model_v1/image_"+Nimg+".tif", cv.IMREAD_GRAYSCALE)
    
    cropped, coords = crop_image(img, return_coords=True)

    x0, x1, y0, y1 = coords
    cropped_mask = mask[y0:y1, x0:x1]

    coord = detect_ap(cropped,9, path = None)
    good = 0
    score = []

    size = len(coord["superficial"]["x"])
    for i in range(size):
        x = int(coord["superficial"]["x"][i])
        y = int(coord["superficial"]["y"][i])
        x_min = max(0, x-1)
        x_max = min(cropped_mask.shape[0], x+2)
        y_min = max(0, y-1)
        y_max = min(cropped_mask.shape[1], y+2)

        if np.any(cropped_mask[x_min:x_max, y_min:y_max] > 200):
            good += 1
    score.append(good / size *100) #score for superficial aponeurosis

    good = 0
    size = len(coord["deep"]["x"])
    for i in range(size):
        x = int(coord["deep"]["x"][i])
        y = int(coord["deep"]["y"][i])
        x_min = max(0, x-1)
        x_max = min(cropped_mask.shape[0], x+2)
        y_min = max(0, y-1)
        y_max = min(cropped_mask.shape[1], y+2)

        if np.any(cropped_mask[x_min:x_max, y_min:y_max] > 200):
            good+=1
    
    score.append(good / size * 100) #score for deep aponeurosis
    score.append((score[0]+score[1])/2) # mean score
    
    print(score)

    return score

verif("0000")
