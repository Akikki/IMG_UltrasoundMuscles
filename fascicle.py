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

