import matplotlib.pyplot as plt
import matplotlib.image as img 
import pandas as pd
import numpy as np

# GrayScale ---

#ncols,nrows 
fig, axs = plt.subplots(1,2)

sample_img = img.imread('pics/emma.jpeg')
axs[0].imshow(sample_img)

# Select everything then only select the first three of RGB 
grey_img = np.dot(sample_img[...,:3], [0.299, 0.587, 0.144])
# Grey colormap as necessary from imshow  
axs[1].imshow(grey_img, cmap=plt.get_cmap('gray'))


plt.show()
