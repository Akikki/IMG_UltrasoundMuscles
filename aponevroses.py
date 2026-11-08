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

    # x = np.arange(w)
    # axs[2].imshow(image, cmap="gray")
    # axs[2].plot(x, models[0].predict_y(x), "r-")
    # axs[2].plot(x, models[1].predict_y(x), "c-")
    # axs[2].set_ylim(h, 0)
    # axs[2].set_title("Aponeurosis")
    # axs[2].axis("off")

    # plt.tight_layout()
    # plt.show()
    
    return {"superficial": p_droites[0], "deep": p_droites[1]}, models

def angle_between_lines(model_a, model_b):
    m1, m2 = model_a.direction, model_b.direction # tan θ = ∣∣∣m1−m2 / 1+m1m2∣∣∣
    haut = abs(m1[0]*m2[1] - m1[1]*m2[0])
    bas = abs(m1[0]*m2[0] + m1[1]*m2[1])
    return np.degrees(np.arctan2(haut, bas))    

def detect_fascicle(Nimg):
    path = f"data/apo_imgs_v1/apo_images_new_model_v1/image_{Nimg}.tif"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    
    cropped, _ = crop_image(img, return_coords=True)
    points, (sup, deep) = detect_ap(cropped, 9, path=path)

    # We are cropping the image once again to keep only the part between the two aponeurosis, this will allow us to find the fascicle more easily
    yMAX= max(points["superficial"]["y"])
    yMIN = min(points["deep"]["y"])

    cropped = cropped[int(yMAX):int(yMIN)]
    img_hessian = ski.filters.hessian(cropped,sigmas=range(2, 8, 2),black_ridges=False) #black ridges because we are using a black font and white lines

    h,w = cropped.shape
    centers, means = [], [] # Tuple
    block = 6
    low_ang, high_ang = 8, 45 # Angles are between 5 to 45 degrees 

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

    parts = []
    # parts.append(coord[cropped[0:h, 0:int(w/3)]])
    # parts.append(coord[cropped[0:h, int(w/3):int(2*w/3)]])
    # parts.append(coord[cropped[0:h, int(2*w/3):w]])
    parts.append(coord[(coord[:, 0] >= 0) &(coord[:, 0] < w/3)])
    parts.append(coord[(coord[:, 0] >= w/3) &(coord[:, 0] < 2*w/3)])
    parts.append(coord[(coord[:, 0] >= 2*w/3) &(coord[:, 0] <= w)])

    def angle_ok(model, *data):
        return low_ang <= angle_between_lines(model, deep) <= high_ang

    points, models, angles = {}, [], []
    for i, part in enumerate(parts):
        model, inliers = ransac(part, LineModelND, min_samples=2, residual_threshold= 3, max_trials=2000, is_model_valid=angle_ok)
        inlier_pts = part[inliers]
        if angle_between_lines(model,deep) < low_ang or angle_between_lines(model,deep) > high_ang:
            print(f"Fascicle {i + 1} rejected: angle {angle_between_lines(model, deep):.1f}°")
            continue
        points[f"fascicle_{i + 1}"] = {"x": inlier_pts[:, 0], "y": inlier_pts[:, 1]} 
        models.append(model)
        angles.append(angle_between_lines(model, deep))  

    names = list(points.keys())      # même ordre que models et angles (on les a ajoutés ensemble)
    if len(angles) >= 3:        
        med = np.median(angles)
        keep = [i for i, a in enumerate(angles) if abs(a - med) <= 5]
        for i, name in enumerate(names):
            if i not in keep:
                del points[name]

        models = [models[i] for i in keep]
        angles = [angles[i] for i in keep]

    # fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    # axs[0].imshow(img, cmap="gray");  
    # axs[0].set_title("Image originale")
    # axs[0].axis("off")

    # axs[1].imshow(img_hessian, cmap="gray");  
    # axs[1].set_title("Image hessian croppé")
    # axs[1].axis("off")

    # x = np.arange(w)
    # axs[2].imshow(img_hessian, cmap="gray")
    # for i in range(len(models)):
    #     axs[2].plot(x, models[i].predict_y(x))
    # axs[2].set_ylim(h, 0)
    # axs[2].set_title("Fascicles")
    # axs[2].axis("off")

    # plt.tight_layout()
    # plt.show()

    moy_ang = np.mean(angles)
    return points, models, moy_ang
    


""" TEST /////////// """
if __name__ == "__main__":
    img = cv.imread("data/apo_imgs_v1/apo_images_new_model_v1/image_0000.tif", cv.IMREAD_GRAYSCALE)
    cropped = crop_image(img)
    detect_ap(cropped,9,path="data/apo_imgs_v1/apo_images_new_model_v1/image_0000.tif")
    
      
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
        print(detect_fascicle("0001"))

