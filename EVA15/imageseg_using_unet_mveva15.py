# -*- coding: utf-8 -*-
"""ImageSeg_using_UNET_MVEVA15.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Yut6l9uDEytqMy4pnUulgMQB2ot4UPl2
"""

import psutil
def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
print("="*40, "Memory Information", "="*40)
svmem = psutil.virtual_memory()
print(f"Total: {get_size(svmem.total)}") ; print(f"Available: {get_size(svmem.available)}")
print(f"Used: {get_size(svmem.used)}") ; print(f"Percentage: {svmem.percent}%")

!nvidia-smi

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

import sys
sys.path.append('/content/gdrive/My Drive/EVA4/Assignment_15/15A/files/')
from imports.imports_eva import *

from overlay import overlay

from dataloaders.custom_data_loader import custom_data_loader

from models.UNet import UNet

DO ALL IMPORTS HERE

import time
import glob
import torch
import os
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

from IPython.display import Image, clear_output 
print('PyTorch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

from models.Net import Net

CHECK THE DATA TO MAKE SURE FOLDERS ARE IN PLACE IN GDRIVE. 
ACROSS MULTIPLE GDRIVES, so sharing options shlud bec checked
check number of images for all base images and zipfile contents for depth images

cd /content/gdrive/My Drive/Assignment15A/mv_data/EVA4_Custom_Data/

from models.CustomNet import CustomNet

!ls '/content/gdrive/My Drive/Assignment15data/assignment15/FGMASK_OVERLAY' | wc -l

classes = []


train_losses = []
test_losses = []
train_acc = []
test_acc = []

FIRST WE USE THE UNET MODEL which is pretty standard for this type of depth estimation
research for Unet - 
http://deeplearning.net/tutorial/unet.html#:~:text=U%2DNet%20is%20a%20Fully,for%20classification%20and%20segmentation%20tasks.

The U-Net architecture is built upon the Fully Convolutional Network and modified in a way that it yields better segmentation. 
Compared to FCN-8, the two main differences are (1) U-net is symmetric and (2) the skip connections between the downsampling path and the upsampling path apply a concatenation operator instead of a sum. 
Unet follows the architecture conv_layer1 -> conv_layer2 -> max_pooling -> dropout(optional) for contractive path and 
conv_2d_transpose -> concatenate -> conv_layer1 -> conv_layer2 for expansive path. 
Hsd to make  changes to take in_channels as 6 channels in this architecture

!pip install torchsummary
from torchsummary import summary

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print (device)

model = UNet(in_channels=6, out_channels=1).to(device)
summary (model, input_size=(6, 224, 224))
print (model)

data_transform = custom_data_loader.get_def_data_transform()

dataset = custom_data_loader.custom_data_set(root_path = '/content/EVA4_Custom_Data/', folder = 'images', transform = data_transform)
dataset_test = custom_data_loader.custom_data_set(root_path = '/content/EVA4_Custom_Data/', folder = 'images', transform = data_transform)

dataset_loader = custom_data_loader.custom_data_loader (dataset, batch_size=128, num_workers=4, shuffle=True)
test_dataset_loader = custom_data_loader.custom_data_loader (dataset_test, batch_size=128, num_workers=4, shuffle=False)

PATH = '/content/gdrive/My Drive/EVA4/Assignment_15/15A/files/Model_UNet_14_06.pth'

#BCE with Logitsloss tested

#criterion = nn.L1Loss().cuda()
#criterion = nn.MSELoss().cuda()
#criterion = nn.CrossEntropyLoss().cuda()
criterion = nn.BCEWithLogitsLoss().cuda()

def train(model, device, train_loader, optimizer, epoch, criterion):
  train_losses = []
  train_acc = []
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0

  for batch_idx, (data) in enumerate(pbar):
    idx = 0
    # get samples
    images = data["Image"].to(device) 
    image_bgs = data["Image_Bg"].to(device) 
    masks = data["Mask"].to(device)

    image_and_bg = torch.cat ((images,image_bgs), dim = 1)

    # Init
    optimizer.zero_grad()

    # Predict
    y_pred = model(image_and_bg)

    # Calculate loss
    criterion = criterion
    loss = criterion(y_pred, masks)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    if batch_idx % 1500 == 0:
      torch.save(model.state_dict(), PATH)

    pbar.set_description(desc= f'Loss={loss.item():0.3f} Batch_id={batch_idx} ')
    train_acc.append(loss)

  return train_losses, train_acc

def test(model, device, test_loader):
  model.eval()
  test_losses = []
  test_acc = []
  test_loss = 0
  correct = 0

  with torch.no_grad():
      for idx, data in enumerate(test_loader):
        images = data["Image"].to(device) 
        image_bgs = data["Image_Bg"].to(device)   
        masks = data["Mask"].to(device)

        image_and_bg = torch.cat ((images,image_bgs), dim = 1)

        output = model(image_and_bg)
        criterion = nn.BCEWithLogitsLoss().cuda()
        test_loss += criterion(output, masks).item()  # sum up batch loss


  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  print('\nTest set: Average loss: {:.4f}, Accuracy: /{} \n'.format(
      test_loss, len(test_loader.dataset)))
  
  test_acc.append(test_loss)
  return test_losses, test_acc

model = CustomNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.000001)  # Removed Weight Decay
scheduler = StepLR(optimizer, step_size=1, gamma=0.01)

from tqdm.notebook import tqdm

EPOCHS = 1
for epoch in range(EPOCHS):
    print("EPOCH:", epoch+1)
    a,b = train(model, device, dataset_loader, optimizer, epoch)
    train_losses.extend(a)
    train_acc.extend(b)

    scheduler.step()
    print('')

PATH = '/content/gdrive/My Drive/EVA4/Assignment_15/15A/files/Model_CustomNet_14_06.pth'

model = CustomNet().to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

images = iter(dataset_loader).next()

image = images["Image"].to(device)
image_bg = images["Image_Bg"].to(device)
masks = images["Mask"].to(device)

image_and_bg = torch.cat ((image, image_bg), dim = 1)

y_pred = model(image_and_bg)

def show_image(inp, n_row=8, title=None, mean=None, std=None):

    inp = torchvision.utils.make_grid(inp.detach().cpu(), n_row)
    inp = inp.numpy().transpose((1, 2, 0))
    if mean:
        mean = np.array(mean)
        std = np.array(std)
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(10, 10))
    plt.savefig('/content/EVA4_Custom_Data/mask_images',inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

show_image(y_pred[::4], n_row=8, title='Predicted (Mask)')

show_image(unet_pred[::4].cpu(), n_row=8, title='Predicted (Mask)')

!pip install torchsummary
from torchsummary import summary

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print (device)

model = Net().to(device)
print (model)

!pip install torchsummary
from torchsummary import summary

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print (device)

model = CustomNet().to(device)
summary (model, input_size=(6, 64, 64))
print (model)