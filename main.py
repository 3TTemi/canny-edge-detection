import matplotlib.pyplot as plt
import matplotlib.image as img 
from scipy import signal
import pandas as pd
import numpy as np

def convert_grayscale(img):
    r_val = 0.299
    g_val = 0.587
    b_val = 0.114

    r = img[...,0]
    g = img[..., 1]
    b = img[..., 2]
    return r_val * r + g_val * g + b_val * b


def gaussian_filter(filter_size, sigma):
    # constant within the formula 

    const = 1 / (2.0 * np.pi * sigma**2)
    iter_size = filter_size // 2

    # Creating the filter, originally empty with 0's 
    filter = np.zeros((filter_size,filter_size), np.float32)

    # Coverts the range of the filter matrix
    for x in range (-iter_size, iter_size):
        for y in range(-iter_size, iter_size):
            exp_val = np.exp(-(x**2.0 + y**2.0) / (2.0 * (sigma **2.0)))
            filter[x + iter_size, y + iter_size] = const * exp_val
    
    return filter


def gradient_sobel_calc(blur_img):
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,-0,1]], np.float32)
    Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)

    Gx_img = signal.convolve2d(blur_img, Gx)
    Gy_img = signal.convolve2d(blur_img, Gy)

    G = np.sqrt((Gx_img ** 2.0)+(Gy_img ** 2.0))
    theta = np.arctan2(Gx_img, Gy_img)
    return (G,theta)
        
def non_max_supression(G, theta):

    row, col = G.shape
    Z = np.zeros((row,col), np.int32)

    angle = theta * 180. / np.pi 
    angle[angle < 0] += 180 

    for i in range (1, row - 1):
        for j in range (1, col - 1):
            q = 255 
            r = 255 

            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                r = [i, j-1]
                q = G[i, j+1]

            elif (22.5 <= angle[i,j] < 67.5):
                r = G[i-1, j+1]
                q = G[i+1, j-1]

            elif (67.5 <= angle[i,j] < 112.5):
                r = G[i-1, j]
                q = G[i+1, j]

            elif (112.5 <= angle[i,j] < 157.5):
                r = G[i+1, j+1]
                q = G[i-1, j-1]

            if ((G[i,j] >= q) and (G[i,j] >= r)).all():
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0

    return Z 

def double_threshold(img, lowRatio, highRatio):

    highThreshold = img.max() * highRatio
    lowThreshold = lowRatio * highRatio

    width, height = img.shape 
    result = np.zeros((width,height), np.int32)

    weak_pixel_val = 75 
    strong_pixel_val = 255 

    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    result[strong_i,strong_j] = strong_pixel_val
    result[weak_i,weak_j] = weak_pixel_val

    return result 

def hysteresis(img, weak, strong):
    for i in range(1,img.shape[0] - 1):
        for j in range(1,img.shape[1] - 1):
             if (img[i, j] == weak):
                if (
                    (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)
                ):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img

import numpy as np

#n cols, n rows subplots 
fig, axs = plt.subplots(1,2)

sample_img = img.imread('pics/emma.jpeg')

grey_img = convert_grayscale(sample_img)

blurred_img =  signal.convolve2d(grey_img, gaussian_filter(15,5))

gradient, theta = gradient_sobel_calc(blurred_img)

suppressed_img = non_max_supression(gradient, theta)

threshold_img = double_threshold(suppressed_img, 0.05, 0.14)

final_img = hysteresis(np.copy(threshold_img), 75, 255)

# Grey colormap as necessary for imshow  
axs[0].imshow(sample_img, cmap=plt.get_cmap('gray'))
axs[1].imshow(final_img, cmap=plt.get_cmap('gray'))

plt.show()
