import torch
import os
import cv2
from PIL import Image
from torchvision import models
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class overlay:
    def __init__():
        super().__init__()
        
    def flip_image_wo_alpha (basepath, newpath):
        file_names = [entry for entry in os.scandir(basepath) if entry.is_file()]
        
        for idx, file_name in enumerate (file_names, start=1):
            image = cv2.imread (file_name.path, cv2.IMREAD_UNCHANGED)
            image = cv2.flip (image, 1)
            
            cv2.imwrite (newpath + file_name.name, image)
            
    def flip_image_alpha (basepath, newpath):
        file_names = [entry for entry in os.scandir(basepath) if entry.is_file()]
        
        for idx, file_name in enumerate (file_names, start=1):
            image = cv2.imread (file_name.path, cv2.IMREAD_UNCHANGED)
            alpha = image[:,:,3]
            image = image[:,:,:3]
            
            image = cv2.flip (image, 1)
            alpha = cv2.flip (alpha, 1)
            result = np.dstack ([image, alpha])
            
            cv2.imwrite (newpath + file_name.name, result)
            
    def overlay_images (bgpath, fgpath, newpath, name_prefix, bg_img_size, fg_img_size):
        bg_file_names = [entry for entry in os.scandir(bgpath) if entry.is_file()]
        fg_file_names = [entry for entry in os.scandir(fgpath) if entry.is_file()]
        
        for fg_images in fg_file_names:
            for idx, bg_images in enumerate (bg_file_names, start=1):
                new_file_name = newpath + '_' + name_prefix + '_fg_' + fg_images.name[:-4] + '_bg_' + bg_images.name[:-4]

                bg_image = Image.open (bg_images.path)

                fg_image = Image.open (fg_images.path)
                
                x = 1
                for j in range (4, 0, -1):
                    for i in range (0, 5):
                        fg_bg_image = bg_image.copy()
                        fg_bg_image.paste(fg_image, (28*i, 28*j), mask=fg_image)
                        fg_bg_image.save (new_file_name + '_{seq}.{suffix}'.format(seq=str(x).zfill(2), suffix = bg_images.name[-3:]))
                        x += 1

    def overlay_mask_images (bgpath, fgpath, newpath, name_prefix, bg_img_size, fg_img_size):
        bg_file_names = [entry for entry in os.scandir(bgpath) if entry.is_file()]
        fg_file_names = [entry for entry in os.scandir(fgpath) if entry.is_file()]

        for fg_images in fg_file_names:
            for idx in range (1, 101):
                new_file_name = newpath + name_prefix + '_mask_' + fg_images.name[:-4] + '_bg_' + bg_file_names[0].name[:-4] + '_' + str(idx).zfill (3)

                bg_image = Image.open (bg_file_names[0].path)

                fg_image = Image.open (fg_images.path)
                
                x = 1
                for j in range (4, 0, -1):
                    for i in range (0, 5):
                        fg_bg_image = bg_image.copy()
                        fg_bg_image.paste(fg_image, (28*i, 28*j))
                        fg_bg_image.save (new_file_name + '_{seq}.{suffix}'.format(seq=str(x).zfill(2), suffix = bg_file_names[0].name[-3:]))
                        x += 1
                        
    def load_all_images (bgpath, fgpath, newpath, name_prefix, bg_img_size, fg_img_size):
        bg_file_names = [entry for entry in os.scandir(bgpath) if entry.is_file()]
        fg_file_names = [entry for entry in os.scandir(fgpath) if entry.is_file()]
        
        for fg_images in fg_file_names:
            for idx, bg_images in enumerate (bg_file_names, start=1):
                new_file_name = newpath + name_prefix + '_' + fg_images.name[:-4] + '_bg_' + bg_images.name[:-4]

                bg_image = Image.open (bg_images.path)
                
                x = 1
                for j in range (4, 0, -1):
                    for i in range (0, 5):
                        fg_bg_image = bg_image.copy()
                        fg_bg_image.save (new_file_name + '_{seq}.{suffix}'.format(seq=str(x).zfill(2), suffix = bg_images.name[-3:]))
                        x += 1

        
 
