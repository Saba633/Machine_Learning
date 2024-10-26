from PIL import Image
from matplotlib import pyplot as plt

img = Image.open(r'D:\Code\Python\Images\cell-division.jpg') #<----------Your image path

plt.imshow(img)
plt.axis('off')
plt.title('single image display')
plt.show()

# ------------------------Multiple Image Display-----------------------

import os

img_path = r'D:\Code\Python\Images' #<----------Your image folder path

read_img = [images for images in os.listdir(img_path) if images.endswith(('png', 'jpg', 'jpeg', 'jfif'))]
fig, axs = plt.subplots(1,5, figsize=(10,6))  #---------- to manage mulitple plots at the same time
for  i, image in enumerate(read_img):
    path = os.path.join(img_path, image)
    opn_img = Image.open(path)
    axs[i].imshow(opn_img)
    axs[i].axis('off')
    axs[i].set_title(f'Image-{i+1}')

plt.show()
