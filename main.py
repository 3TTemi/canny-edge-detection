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




print(sobel_filter(blurred_img))

# axs[0].imshow(grey_img, cmap=plt.get_cmap('gray'))
# axs[1].imshow(blurred_img, cmap=plt.get_cmap('gray'))




# np.convolve(img, )
    

# plt.show()
