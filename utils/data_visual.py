import numpy as np 
import matplotlib.pyplot as plt 
import os

mode = 'train'

depth_path = '../Dataset/CityScapeDepthDataset/' + mode + '/depth/'
image_path = '../Dataset/CityScapeDepthDataset/' + mode + '/image/'

index = 20

depth = np.load(depth_path + str(index) + '.npy')
image = np.load(image_path + str(index) + '.npy')

print(depth.shape)
print(image.shape)

depth = np.moveaxis(depth, -1, 0)

plt.figure(figsize=(18,18))
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(depth[0], cmap='gray')

plt.show()
