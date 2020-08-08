# EVAS14
Objective
Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object.

It is difficult to get this size of a dataset, so we shall create the same :)

**Dataset**
A custom dataset will be used to train this model, which consists of:

100 background images; 100 foreground images

background images are streets and natural artifacts - river, hill, valley etc.

foreground images are cats, dogs, cats, people, cats and dogs together.

**Pre-processing tasks**

Fg transparency - The foreground images were downloaded as transparent from chrome and bing images with setting = transparent ; remove-bg online tool as well as using powerpoint to remove bg helped.

Masks for Foreground - this was done in GIMP by removing the alpha channel and merging the layers. My daughter helped me for a nominal payment.


**1.2 million dataset is created as follows**

*400k foreground overlayed on background images*

100 fg images (in png to preserve transparency, 96X96 size) overlayed on top of 100 bg images (in jpg; size 192X192) in 20 random positions -> this gives rise to 2,00,000 images

100 flipped fg images (in png to preserve transparency, 96X96 size) overlayed on top of 100 bg images (in jpg; size 192X192) in 20 random positions -> this gives rise to 2,00,000 images

*400k (white, non alpha) mask foreground overlayed on black background images -  simulataneously in the same 20 random locations*

100 masked fg images (in png to preserve transparency, 96X96 size) overlayed on top of 100 bg images (in jpg; size 192X192) in 20 random positions -> this gives rise to 2,00,000 images

100 flipped masked fg images (in png to preserve transparency, 96X96 size) overlayed on top of 100 bg images (in jpg; size 192X192) in 20 random positions -> this gives rise to 2,00,000 images

--total til now 8,00,000

*400k depth maps for the foreground overlayed on background images*

Depth images are created by using the mask on bg images (unflipped and flipped) as well as the fg on bg images (unflipped and flipped) for each correcponsing random location. 

**Dataset Samples**

Background: 
![Background](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/bg1.jpg)

![Background series](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/bg_images.png)

Foreground-Background: 

![FG_BG](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/1FGFLIP_BG_flip_fg_fg001_bg_bg1_01.jpg)

Foreground-Background Mask: 

![Mask](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/fg-001-mask.png)

![Mask Flip](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/1FGMFflip_mask_fg001-mask_bg_black_image192_001_01.jpg)

![Mask w/o Flip](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/1FG_MASKbase_mask_fg001-mask_bg_black_image192_001_01.jpg)

![Mask examples](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/Fg-bg-mask.png)


Foreground-Background Depth: 

![background](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/bg94.jpg)

![Depth](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/fg021flip_bg94_05.jpg)

![Depth2] (https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/fg021flip_bg94_08.jpg)

![depth examples](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/Depth-masks.png)

**Notations**

Background image: bg

Foregroung overlayed on background: fg_bg

Mask for fg_bg: fg_bg_mask

Depth map for fg_bg: fg_bg_depth

Mask prediction: mask_pred

Depth map prediction: depth_pred

All Images are available in this link https://drive.google.com/drive/folders/1E8V1RUeF--_THXxJ06j7L4vYxOGnlVIk?usp=sharing

**learnings**

1. OOM is a constant problem in depth models. 

2. fg_bg overlay took about 3-4 hrs to run for each set of 2,00,000 images. total of 4 sets here, so nearly 16/20 hours of effort on this alone.

3. folder naming had to be done very carefully. Lost lot of time due to not organising folders properly, learnt that fast, and split fg images into 3 sets of 37, 39 and 24. once organised this way, the processing ran faster.

4. gdrive has a 15GB free limit - it is sacrosanct, so need to share colab files across gmail ids (i am the proud owner of 5 gmail id's now all for this assignment)

5. zipfile is beautiful to use but one cannot unzip the file and view the pictures ; imshow with zipfile is patchy- so run a small batch of 24 first, then make sure entire process works before running for all images.

total time taken - fg/bg ; flip fg/bg, mask fg_bg and flip_mask fg_bg --> total 16 hrs

Depth model - 21 hours in total. 



