import matplotlib.pyplot as plt
import matplotlib.image as img 
from scipy import signal
import pandas as pd
import numpy as np

# GrayScale ---

#ncols,nrows 
fig, axs = plt.subplots(1,2)

sample_img = img.imread('pics/emma.jpeg')
# axs[0].imshow(sample_img)

# Select everything then only select the first three of RGB 
grey_img = np.dot(sample_img[...,:3], [0.299, 0.587, 0.144])
# Grey colormap as necessary from imshow  
# axs[1].imshow(grey_img, cmap=plt.get_cmap('gray'))


# Gausian Filterring ----

def gaussian_filter(filter_size, sigma):
    # constant within the formula 

    const = 1 / (2.0 * np.pi * sigma**2)
    iter_size = filter_size // 2

    # Creating the filter, originally empty with 0's 
    filter = np.zeros((filter_size,filter_size), np.float32)

    # Coverts the range of the filter matrix
    for x in range (-iter_size, iter_size + 1):
        for y in range(-iter_size, iter_size + 1):
            exp_val = np.exp(-(x**2.0 + y**2.0) / (2.0 * (sigma **2.0)))
            filter[x + iter_size, y + iter_size] = const * exp_val
    
    return filter

# print(gaussian_filter(5,1).shape)
# blurred_img =  signal.fftpack.fft2(gaussian_filter(5,1.4),grey_img, axes=(0,1))
kernel = gaussian_filter(5,1.4)
blurred_img =  signal.convolve2d(grey_img, gaussian_filter(5,1.4))
# blurred_img =  np.convolve(grey_img, kernel, mode='same') / sum(kernel)

def sobel_filter(blur_img):
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,-0,1]], np.float32)
    Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)

    Ix = signal.convolve2d(blur_img, Gx)
    Iy = signal.convolve2d(blur_img, Gy)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (G,theta)
        

gradient, theta = sobel_filter(blurred_img)

def no_max_supression(G, theta):

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

suppressed_img = no_max_supression(gradient, theta)

# axs[0].imshow(grey_img, cmap=plt.get_cmap('gray'))


def double_threshold(img, lowRatio, highRatio):

    highThreshold = img.max() * highRatio
    lowThreshold = lowRatio * highRatio

    width, height = img.shape 
    result = np.zeros((width,height), np.int32)

    weak_pixel_val = 30 
    strong_pixel_val = 255 

    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    result[strong_i,strong_j] = strong_pixel_val
    result[weak_i,weak_j] = weak_pixel_val

    return result 

threshold_img = double_threshold(suppressed_img, 0.07, 0.1)

axs[0].imshow(threshold_img, cmap=plt.get_cmap('gray'))


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

final_img = hysteresis(threshold_img, 30, 255)

axs[1].imshow(final_img, cmap=plt.get_cmap('gray'))


plt.show()
