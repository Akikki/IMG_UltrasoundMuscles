import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path
from skimage.measure import ransac, LineModelND
from resize import main_region, crop_image
from aponevroses import detect_ap
import skimage as ski

""" In this code we will calulate the Pennation Angle (PA), the Fascicle Length (FL) and the Muscle Thickness (MT)"""

# def calculate_MT(Nimg):
#     img = cv.imread("data/apo_imgs_v1/apo_images_new_model_v1/image_"+Nimg+".tif", cv.IMREAD_GRAYSCALE)
#     #mask = cv.imread("data/apo_masks_v1/apo_masks_new_model_v1/image_"+Nimg+".tif", cv.IMREAD_GRAYSCALE)
#     cropped = crop_image(img, return_coords=True)
#     coords = detect_ap(cropped,9, path = None)

#     x_sup, y_sup = coords["superficial"]["x"],coords["superficial"]["y"]
#     x_deep, y_deep = coords["deep"]["x"],coords["deep"]["y"]

#     dis = []
#     for i in range(len(x_sup)):
#         if x_sup[i] == x_deep[i]:
#             dis.append(y_deep[i] - y_sup[i])

#     moy = np.mean(dis)
#     print("Muscle Thickness (MT) for image "+Nimg+" in px : ", moy)

#     return moy 

def calculate_MT(Nimg):
    path = f"data/apo_imgs_v1/apo_images_new_model_v1/image_{Nimg}.tif"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    cropped, _ = crop_image(img, return_coords=True)
    points, (sup, deep) = detect_ap(cropped, 9, path=path)

    # We calculate the distance between the two lines where the x values are the same 

    x0 = max(points["superficial"]["x"].min(), points["deep"]["x"].min())
    x1 = min(points["superficial"]["x"].max(), points["deep"]["x"].max())
    x = np.arange(int(x0), int(x1) + 1)

    moy = np.mean(deep.predict_y(x) - sup.predict_y(x))
    print(f"Muscle Thickness (MT) pour l'image {Nimg} est de {moy:.1f} px")

    return moy

if __name__ == "__main__":
    Nimg = "0000"
    calculate_MT(Nimg)

    path = f"data/apo_imgs_v1/apo_images_new_model_v1/image_{Nimg}.tif"
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)

    plt.imshow(img)
    plt.title('Image test')
    plt.axis('off')
    plt.show()