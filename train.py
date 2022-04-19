import argparse

from torchvision.transforms.transforms import Normalize
from model import PSPNet
import torch.nn as nn 
import torch.nn.functional as F 
from torch import optim 
import math 
import torch
import torch.utils.data as data 
import time 
from depth_data import CityScapes
from torchvision import transforms

def args_input():
    parser =argparse.ArgumentParser(description="Set up parameters")

    parser.add_argument('--batch_size', type=int, nargs='?', default=32 ,help='batch size')
    parser.add_argument('--num_epochs', type=int, nargs='?', default=10000, help='number of epochs')
    parser.add_argument('--aux_weight', type=float, nargs='?', default=0.4, help='aux weight for aux loss [0, 1]')
    parser.add_argument('--scratch', type=bool, nargs='?', default=False, help='Train from scratch ?')
    return parser

class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        loss = F.mse_loss(outputs[0], targets, reduction='mean')
        loss_aux = F.mse_loss(outputs[1], targets, reduction='mean')

        return loss + self.aux_weight*loss_aux

def lambda_epoch(epoch):
    max_epoch = 10000
    return math.pow(1-epoch/max_epoch, 0.9)

def train_model(model, dataloader, criterion, scheduler, optimizer, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    model.to(device)

    num_train_imgs = len(dataloader['train'].dataset)
    num_val_imgs = len(dataloader['val'].dataset)
    batch_size = dataloader['train'].batch_size

    print('num train imgs: ', num_train_imgs)
    print('num val imgs', num_val_imgs)

    iteration = 1
    
    min_loss = 0

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print('Epoch {} / {}'.format(epoch+1, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                optimizer.zero_grad()
                # scheduler.step()
                print('Train')
            else:
                  model.eval()
                  print('Val')
            
            count = 0
            for images, depth_images in dataloader[phase]:
                if images.size()[0] == 1:
                    print('image size [0] = 1 !!!')
                    continue
                images = images.to(device)
                depth_images = depth_images.to(device)

                if (phase == 'train') and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    
                    loss = criterion(outputs, depth_images)

                    if phase == 'train':
                        loss.backward()

                        if (iteration % 10 == 0):
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print('Iteration {} || Loss: {:.6f}  || 10iter: {:.6f} sec'.format(iteration, loss.item()/batch_size, duration))

                            t_iter_start = time.time()
                        
                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()

            if phase == 'train':
              scheduler.step()
        
        t_epoch_end = time.time()
        duration = t_epoch_end - t_epoch_start
        print('Epoch {} || Epoch_train_loss: {:.6f} || Epoch_val_loss: {:.6f}'.format(epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs))        
        print('Duration {:.6f} sec'.format(duration))
        t_epoch_start = time.time()
        if min_loss == 0:
          min_loss = epoch_val_loss

        elif epoch_val_loss < min_loss and min_loss != 0:
            min_loss = epoch_val_loss
            print('Saved model')
            # torch.save(model.state_dict(), 'Weights/pspnet50_new_1.pth')

if __name__ == '__main__':
    parser = args_input().parse_args()


    root_path = 'CityScapeDepthDataset'
    train_dataset = CityScapes(root_path, phase='train')
    val_dataset = CityScapes(root_path, phase='val')

    model = PSPNet(n_classes=1)

    criterion = PSPLoss(aux_weight=parser.aux_weight)

    optimizer = optim.Adam([
        {'params': model.feature_conv.parameters(), 'lr': 1e-3},
        {'params': model.feature_res_1.parameters(), 'lr': 1e-3},
        {'params': model.feature_res_2.parameters(), 'lr': 1e-3},
        {'params': model.feature_dilated_res_1.parameters(), 'lr': 1e-3},
        {'params': model.feature_dilated_res_2.parameters(), 'lr': 1e-3},
        {'params': model.pyramid_pooling.parameters(), 'lr': 1e-3},
        {'params': model.decode_feature.parameters(), 'lr': 1e-3},
        {'params': model.aux.parameters(), 'lr': 1e-3},
    ], weight_decay=0.00001)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    batch_size = parser.batch_size

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloader = {
        'train': train_loader,
        'val': val_loader
    }

    num_epochs = parser.num_epochs
    train_model(model, dataloader, criterion, scheduler, optimizer, num_epochs=num_epochs)