import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
from pathlib import Path
from skimage.measure import ransac, LineModelND
import os
from resize import main_region, crop_image
import skimage as ski

""" PARTIE 2 --------------------------------------------------------------------------------------------------"""

def detect_ap(image,block,path):
    if path == None :
        path = "No path"
    h, w = image.shape  

    img_hessian = np.percentile(image,95) #partie de l'image la plus lumineuse.
    img_hessian = ski.filters.hessian(image,sigmas=range(2, 8, 2),black_ridges=False) #black ridges because we are using a black font and white lines

    #COPY = image.copy() # We will draw our points on this image
    centers, means = [], [] # Tuple
    
    for i in range(0, h - block + 1, block):          
        for j in range(0, w - block + 6, block):      
            bloc = img_hessian[i:i + block, j:j + block+5]
            means.append(bloc.mean())
            centers.append((j + block // 2, i + block // 2))
    means = np.array(means)
    centers = np.array(centers)

    keep = means < np.percentile(means, 18)
    coord = centers[keep]
    nbr_points = len(coord)

    while nbr_points > 1000: 
        print("More than 1000 points, will be unaccurate", nbr_points)
        centers, means = [], []
        block+= int(0.001*nbr_points)
        for i in range(0, h - block + 1, block):          
            for j in range(0, w - block + 1, block):      
                bloc = img_hessian[i:i + block, j:j + block]
                means.append(bloc.mean())
                centers.append((j + block // 2, i + block // 2))
        means = np.array(means)
        centers = np.array(centers)
        keep = means < np.percentile(means, 18)
        coord = centers[keep]
        nbr_points = len(coord)
    
    #Now we are going to cut the image in half to apply the RANSAC algorithm to both images, allowing us to find the two aponeurosis.

    y_split = np.median(coord[:, 1]) # We are not cutting the image in half it's height because we don't know where the aponeurosis is, instead we split it based on the median of the points
    halves = [coord[coord[:, 1] < y_split], coord[coord[:, 1] >= y_split]]

    models ,p_droites= [],[]
    for pts in halves:
        model, inliners = ransac(pts.astype(float), LineModelND, min_samples=2, residual_threshold=block, max_trials=1000)
        models.append(model)

        inlier_pts = pts[inliners]               # coordonnées (x, y) des points retenus par RANSAC
        
        #x = np.arange(w)
        #p_droites.append({"x": x, "y": model.predict_y(x) })

        p_droites.append({"x": inlier_pts[:, 0],"y": inlier_pts[:, 1]})

    # fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    # axs[0].imshow(img_hessian, cmap="gray");  
    # axs[0].set_title("Image hessian")
    # axs[0].axis("off")
    # axs[1].imshow(image, cmap="gray")
    # axs[1].scatter(coord[:, 0], coord[:, 1], c="b", s=9)
    # axs[1].set_title(os.path.basename(path))
    # axs[1].axis("off")

    # x = np.arange(cropped.shape[1])
    # axs[2].imshow(cropped, cmap="gray")
    # plt.plot(x, models[0].predict_y(x), "r-")
    # plt.plot(x, models[1].predict_y(x), "c-")
    # plt.ylim(cropped.shape[0], 0)
    # axs[2].set_title("Aponeurosis")
    # axs[2].axis("off")

    # plt.tight_layout()
    # plt.show()
    
    return {"superficial": p_droites[0], "deep": p_droites[1]}

folder = Path("data/apo_imgs_v1")
images = sorted(
    p for p in folder.rglob("*")
    if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg")
)

random.seed(100)
selected = random.sample(images, 1)

for i, path in enumerate(selected):
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    cropped = crop_image(img)
    print(detect_ap(cropped,9,path))
