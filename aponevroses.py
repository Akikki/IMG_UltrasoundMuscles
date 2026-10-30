from email.mime import image

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
from pathlib import Path
from skimage.measure import ransac, LineModelND

from resize import main_region, crop_image
import skimage as ski

""" PARTIE 2 --------------------------------------------------------------------------------------------------"""

def detect_ap(image,block):
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

    models = []
    model, _ = ransac(coord.astype(float), LineModelND, min_samples=2, residual_threshold=block/2, max_trials=1000)
    models.append(model)

    print("Nombres de points trouvés ;",len(coord))

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    axs[0].imshow(
    img_hessian,
    cmap="gray",
    vmin=0,
    vmax=np.percentile(img_hessian, 99)
)
    axs[0].set_title("Image filtrée")
    axs[0].axis("off")
    axs[1].imshow(image, cmap="gray")
    axs[1].scatter(coord[:, 0], coord[:, 1], c="r", s=8)
    axs[1].set_title("Image : ")
    axs[1].axis("off")
    #Resultat final :
    x = np.arange(cropped.shape[1])
    axs[2].imshow(cropped, cmap="gray")
    plt.plot(x, models[0].predict_y(x), "r-")
    plt.plot(x, models[1].predict_y(x), "c-")
    plt.ylim(cropped.shape[0], 0)
    axs[2].set_title("Aponeurosis")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
    
    #return coord, img_hessian, models #sup deep dans models
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
    detect_ap(cropped,9)
