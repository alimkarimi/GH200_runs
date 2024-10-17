import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import os
import numpy as np

def plot_images_from_each_cat():
    ### Plot images from each category ###
    catNms=['airplane','bus','cat', 'dog', 'pizza']
    fig, ax = plt.subplots(5,3, figsize=(16, 16))
    for n in range(0,5):
        temp_dir = os.listdir('train/' + catNms[n])
        #open image
        for i in range(6,9):
            temp_img = Image.open('train/' + catNms[n] + '/' + temp_dir[i]) #get the i-th image from the directory. 
    #         Here, we just grab the 0th, 1st, and 2nd.
            #convert to numpy array for plotting
            temp_np_arr = np.array(temp_img)
            ax[n,i-6].imshow(temp_np_arr)
            ax[n,i-6].set_title(catNms[n] + ' example ' + str(i-5) )

    plt.savefig('example_training_images.jpg')

if __name__ == "__main__":
    plot_images_from_each_cat()
