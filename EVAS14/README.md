# EVAS14
Objective
Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object.

Dataset
A custom dataset will be used to train this model, which consists of:

100 background images
400k foreground overlayed on background images
400k masks for the foreground overlayed on background images
400k depth maps for the foreground overlayed on background images


Dataset Creation: 

Dataset Samples
Background: 

Foreground-Background: 

Foreground-Background Mask: 

Foreground-Background Depth: 

Notations
Background image: bg
Foregroung overlayed on background: fg_bg
Mask for fg_bg: fg_bg_mask
Depth map for fg_bg: fg_bg_depth
Mask prediction: mask_pred
Depth map prediction: depth_pred
