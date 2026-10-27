import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread("apo_imgs_v1/apo_images_new_model_v1/image_0000.tif", cv.IMREAD_GRAYSCALE)

# plt.imshow(img,cmap='gray')
# plt.title('image noise.tif')
# plt.axis('off')
# plt.show()

def resize_image(image):
    """
    Resize the input image into a studiable image. To do this we will start by calculating the mean color and the standard deviation of the image at his center, 
    since the center of the image is usally part of what we will keep. Then we will use the mean color and the standard deviation to clean the image and keep only the pixels that
    do not vary too much. Keeping only the image that we want to study and getting rid of the useless borders.
    """
    height, width = image.shape
    print(height,width)

    # Calculate the mean color and the standard deviation of the image at its center
    h_margin = height//10
    w_margin = width//10
    center_region = image[height//2 - h_margin : height//2 + h_margin, width//2 - w_margin : width//2 + w_margin]
    
    mean_color = np.mean(center_region)
    standard_deviation = np.std(center_region)

    print(mean_color)
    print(standard_deviation)

    # Limitations between two pixels colors, will allow us to create a mask.
    
    lower_bound = mean_color - 0.7*standard_deviation
    upper_bound = mean_color + 0.7*standard_deviation

    """
    We will try to find two corners of the IRM, the first one being the top left corner and the second the bottom right corner. This way we will crop the image with the information of those two points.
    To get the top left corner we look at all the pixels from the top left to the top right then go down a "line", when we find a pixel that is inside the mask, we assume it's the top left corner.
    Then we will run a second double loops starting from the bottom right and going to the left, then up. Again when we find a pixel that is inside the mask, we assume that it's the bottom left corner.
    Then we crop the image knowing the position of those two corners.
    """

    pos_tl = [] # Position of the top left corner. 
    found = False

    i = 0
    while i < height and not found :
        j = 0
        while j < width and not found:
            print("img i,j :" , image[i,j])
            if image[i, j] > lower_bound and image[i, j] < upper_bound:
                found = True
                pos_tl = [i,j]
                print("Top left corner found, position :", pos_tl)
            j += 1
        i += 1

    #cropped_image = image[pos_tl[1]:width, pos_tl[0]:height]

    pos_br = [] # Position of the bottom right corner.
    found = False

    #height, width = cropped_image.shape
    i = height - 1
    while i > 0 and not found:
        j = width -1
        while j > 10 and not found :
            if image[i, j] > lower_bound and image[i, j] < upper_bound:
                pos_br = [i,j]
                print("Bottom right corner found, position :", pos_br)
                found = True
            j -= 1
        i -= 1

    if pos_tl == [0,0] and pos_br == [height-1, width-1]: # At this moment the image is correctly cropped, nothing else has to be deleted from the image.
        return image
    
    cropped = image[pos_tl[0]:pos_br[0]+1 ,pos_tl[1]:pos_br[1]+1]
    return resize_image(cropped) # Recursive until we deleted everyhting useless from the image

cropped = resize_image(img)

plt.imshow(img,cmap='gray')
plt.title('image 0630.tif')
plt.axis('off')
plt.show()

plt.imshow(cropped)
plt.title('image 0630.tif cropped')
plt.axis('off')
plt.show()
