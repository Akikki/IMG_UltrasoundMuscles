from email.mime import image
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
from pathlib import Path
from skimage.measure import ransac, LineModelND

from resize import main_region, crop_image
import skimage as ski

""" PARTIE 2 --------------------------------------------------------------------------------------------------"""

def detect_ap(image,block,n_lines=20):
    h, w = image.shape

    img_hessian = np.percentile(image,95) #partie de l'image la plus lumineuse.
    img_hessian = ski.filters.frangi(
        image,
        sigmas=np.arange(3, 12, 2),
        beta=0.3,
        gamma=None,
        black_ridges=False,
        mode="reflect"
    )

    #COPY = image.copy() # We will draw our points on this image
    centers, means = [], [] # Tuple
    
    for i in range(0, h - block + 1, block):          
        for j in range(0, w - block + 1, block):      
            bloc = img_hessian[i:i + block, j:j + block]
            means.append(np.percentile(bloc, 95))
            centers.append((j + block // 2, i + block // 2))
    means = np.array(means)
    centers = np.array(centers)

    keep = means > np.percentile(means, 80)
    coord = centers[keep]
    remaining = coord.copy() # On debute avec tout les points

    models = []
    candidates = []
    for i in range(n_lines):
        if len(remaining) < 2:  # On peut plus faire de droite
            break

        model, inliers = ransac(
            remaining, # On prend le reste des points de chaque modelisation
            LineModelND,
            min_samples=2,
            residual_threshold=block / 2,
            max_trials=1000,
        )

        if model is None or inliers is None: # Tout est vides = erreur
            break

        candidates.append(model)
        remaining = remaining[~inliers]

# Recherche de la meilleure paire selon la densité de points
    best_pair = None
    best_score = -1
    best_count = 0

    x1, x2 = 0, w - 1
    x = np.arange(x1, x2 + 1)

    for m1, m2 in combinations(candidates, 2):

        # On ignore les droites verticales
        if abs(m1.params[1][0]) < 0.1 or abs(m2.params[1][0]) < 0.1:
            continue

        # On ignore les droites qui se croisent dans l'image
        d = m1.predict_y(np.array([x1, x2])) - m2.predict_y(np.array([x1, x2]))

        if d[0] * d[1] <= 0:
            continue

        # On ignore les droites trop proches
        if np.min(np.abs(d)) < 2 * block:
            continue

        # On compte les points entre les deux droites
        y1 = m1.predict_y(coord[:, 0])
        y2 = m2.predict_y(coord[:, 0])

        inside = (
            (coord[:, 1] > np.minimum(y1, y2)) &
            (coord[:, 1] < np.maximum(y1, y2))
        )

        count = np.sum(inside) # Nb de points entre les 2

        # On calcule la surface entre les droites dans l'image
        y1 = m1.predict_y(x)
        y2 = m2.predict_y(x)

        top = np.clip(np.minimum(y1, y2), 0, h)
        bottom = np.clip(np.maximum(y1, y2), 0, h)

        area = np.sum(bottom - top) # Surface entre les 2

        if area == 0:
            continue

        # Score : densité de points entre les droites
        score = count / (area **0.6)

        if score > best_score:
            best_score = score
            best_count = count
            best_pair = (m1, m2)

    if best_pair is None:
        print("Aucune paire valide trouvée")
        return False

    # Classement des droites de haut en bas
    m1, m2 = best_pair

    y1 = m1.predict_y(np.array([w / 2]))[0]
    y2 = m2.predict_y(np.array([w / 2]))[0]

    if y1 < y2:
        models = [m1, m2]
    else:
        models = [m2, m1]

    print("Nombre de droites candidates :", len(candidates))
    print("Nombre de points entre les droites :", best_count)
    print("Score de densité :", best_score)

    # Affichage des résultats
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(
        img_hessian,
        cmap="gray",
        vmin=0,
        vmax=np.percentile(img_hessian, 99)
    )
    axs[0].set_title("Image filtrée")

    axs[1].imshow(image, cmap="gray")
    axs[1].scatter(coord[:, 0], coord[:, 1], c="r", s=8)
    axs[1].set_title(f"Points sélectionnés ({len(coord)})")

    axs[2].imshow(image, cmap="gray")

    # Affichage des droites candidates en pointillés
    for model in candidates:
        if abs(model.params[1][0]) >= 0.1:
            axs[2].plot(
                x, model.predict_y(x),
                "--", color="orange", alpha=0.5
            )

    # Affichage de la zone entre les deux droites retenues
    y1 = models[0].predict_y(x)
    y2 = models[1].predict_y(x)

    axs[2].fill_between(x, y1, y2, color="yellow", alpha=0.15)
    axs[2].plot(x, y1, "r-", linewidth=2)
    axs[2].plot(x, y2, "c-", linewidth=2)

    axs[2].set_ylim(h, 0)
    axs[2].set_title(f"Aponévroses - score : {best_score:.4f}")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return True

# img = cv.imread("data/apo_imgs_v1/apo_images_new_model_v1/image_0000.tif", cv.IMREAD_GRAYSCALE)
# cropped = crop_image(img)
# detect_ap(cropped,10)

folder = Path("data/test_images_v2")
images = sorted(
    p for p in folder.rglob("*")
    if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg")
)

random.seed(100)
selected = random.sample(images, 10)

for i, path in enumerate(selected):
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    cropped = crop_image(img)
    detect_ap(cropped,9,20)
