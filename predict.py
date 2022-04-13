from model import PSPNet
import torch
import torch.nn as nn 
import numpy as np 
from depth_data import CityScapes
import matplotlib.pyplot as plt

root_path = '../Dataset/CityScapeDepthDataset/'
val_data = CityScapes(root_path, phase='val')

index = 200
image, depth = val_data.__getitem__(index)

model = PSPNet(n_classes=1)
model.load_state_dict(torch.load('Weights/pspnet50_2.pth', map_location=torch.device('cpu')))
model.eval()

inputs = image.detach().numpy()
inputs = np.moveaxis(inputs, 0, -1)

outputs = model(image[None, ...])
outputs = outputs[0].detach().numpy()

real_depth = depth.detach().numpy()

plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.imshow(inputs)
plt.subplot(1,3,2)
plt.imshow(outputs[0][0])
plt.subplot(1,3,3)
plt.imshow(depth[0])

plt.show()