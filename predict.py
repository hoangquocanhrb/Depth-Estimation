import argparse

from model import PSPNet
import torch
import torch.nn as nn 
import numpy as np 
from depth_data import CityScapes
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Choose index of val images")
parser.add_argument('--index', type=int, nargs='?', default=1, help='0-499')

root_path = '../Dataset/CityScapeDepthDataset/'
val_data = CityScapes(root_path, phase='val')

index = parser.index
image, depth = val_data.__getitem__(index)

model = PSPNet(n_classes=1)
model.load_state_dict(torch.load('../Dataset/Weights/pspnet50_new_1.pth', map_location=torch.device('cpu')))
model.eval()

inputs = image.detach().numpy()
inputs = np.moveaxis(inputs, 0, -1)

outputs = model(image[None, ...])
outputs = outputs[0].detach().numpy()

real_depth = depth.detach().numpy()

plt.figure(figsize=(15,15))
plt.subplot(1,3,1)
plt.title('Origin', loc='center')
plt.imshow(inputs.astype('uint8'), vmin=0, vmax=255)
plt.subplot(1,3,2)
plt.title('Predict', loc='center')
plt.imshow(outputs[0][0])
plt.subplot(1,3,3)
plt.title('Ground truth', loc='center')
plt.imshow(depth[0])

plt.show()