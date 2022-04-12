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

        depth = torch.from_numpy(np.moveaxis(depth, -1, 0)) #(1,128,256)
        image = torch.from_numpy(np.moveaxis(image, -1, 0)) #(3,128,256)
        
        return image, depth

    def __len__(self):
        return len(os.listdir(self.datapath + '/image'))

# if __name__ == "__main__":
#     root = '../Dataset/CityScapeDepthDataset/'
#     color_mean = (0.485, 0.456, 0.406)
#     color_std = (0.229, 0.224, 0.225)
#     train_data = CityScapes(root)
#     val_data = CityScapes(root, phase='val')
    

    