import torch 
import numpy as np 
import os 
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class CityScapes(Dataset):
    def __init__(self, root, phase='train'):
        self.root = root 
        self.phase = phase

        if self.phase=='train':
            self.datapath = self.root + '/train'
        else:
            self.datapath = self.root + '/val'

    def __getitem__(self, index):
        image = np.load(self.datapath + '/image/' + str(index) + '.npy')
        depth = np.load(self.datapath + '/depth/' + str(index) + '.npy')

        image = np.moveaxis(image, -1, 0)
        depth = depth*255
        image[0] *= 255
        image[1] *= 255
        image[2] *= 255

        depth = torch.from_numpy(np.moveaxis(depth, -1, 0)) #(1,128,256)
        image = torch.from_numpy(image) #(3,128,256)
        
        return image.float(), depth.float()

    def __len__(self):
        return len(os.listdir(self.datapath + '/image'))

if __name__ == "__main__":
    root = 'CityScapeDepthDataset'
    
    train_data = CityScapes(root, phase='train')
    print(train_data.__len__())
    a, b = train_data.__getitem__(2975)
    

    