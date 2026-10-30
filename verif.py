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

def verif(Nimg):
    img = cv.imread("data/apo_imgs_v1/apo_images_new_model_v1/image_"+Nimg+".tif", cv.IMREAD_GRAYSCALE)
    mask = cv.imread("data/apo_masks_v1/apo_masks_new_model_v1/image_"+Nimg+".tif", cv.IMREAD_GRAYSCALE)

    cropped, coords = crop_image(img, return_coords=True)

    x0, x1, y0, y1 = coords
    cropped_mask = mask[y0:y1, x0:x1]
    coord = detect_ap(cropped,9, path = None)
    score = []
    
    x, y = coord["superficial"]["x"],coord["superficial"]["y"]
    good = 0
    size = len(x)
    for i in range(size):
        ligne = int(y[i])
        colonne = int(x[i])

        x_min = max(0, ligne-25)
        x_max = min(cropped_mask.shape[0], ligne+26)
        y_min = max(0, colonne-25)
        y_max = min(cropped_mask.shape[1], colonne+26)

        if np.any(cropped_mask[x_min:x_max, y_min:y_max] > 200):
            good += 1

    score.append(good / size * 100)


    x, y = coord["deep"]["x"],coord["deep"]["y"]
    good = 0
    size = len(x)

    for i in range(size):
        ligne = int(y[i])
        colonne = int(x[i])

        x_min = max(0, ligne-25)
        x_max = min(cropped_mask.shape[0], ligne+26)
        y_min = max(0, colonne-25)
        y_max = min(cropped_mask.shape[1], colonne+26)

        if np.any(cropped_mask[x_min:x_max, y_min:y_max] > 200):
            good += 1
    score.append(good / size * 100) #score for deep aponeurosis
    
    score.append((score[0]+score[1])/2) # mean score

    # plt.figure(figsize=(12, 7))
    # plt.imshow(cropped_mask, cmap="gray")

    # plt.scatter(coord["superficial"]["x"], coord["superficial"]["y"], c="blue",s=8,label="superficial")
    # plt.scatter(coord["deep"]["x"], coord["deep"]["y"], c="red", s=8, label="Deep")

    # plt.title(f"mask image {Nimg} + pts aponoreusis deep n superficial")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    print(score)
    
    return score

all_scores = []

for i in range(1048):
    Nimg = f"{i:04d}"
    print(f"Image {Nimg}...")

    try:
        score = verif(Nimg)
        all_scores.append(score)

    except Exception as e:
        print(f"Erreur sur l'image {Nimg} : {e}")

# Calcul des moyennes
all_scores = np.array(all_scores)

mean_superficial = np.mean(all_scores[:, 0])
mean_deep = np.mean(all_scores[:, 1])
mean_global = np.mean(all_scores[:, 2])

print("\n===== SCORES MOYENS =====")
print(f"Superficielle : {mean_superficial:.2f} %")
print(f"Profonde      : {mean_deep:.2f} %")
print(f"Moyenne globale : {mean_global:.2f} %")
print(f"Nombre d'images traitées : {len(all_scores)} / 1048")