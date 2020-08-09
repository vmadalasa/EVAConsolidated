# EVAS15 - Image Segmentation


Objective
Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object.

We created the dataset as part of Assignment14, and the statistics were also calculated (mean, stddev for each of the datasets) as part of Assignment 14/15A

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

![Depth2](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/fg021flip_bg94_08.jpg)

![depth examples](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/Depth-masks.png)

**Notations**

Background image: bg

Foregroung overlayed on background: fg_bg

Mask for fg_bg: fg_bg_mask

Depth map for fg_bg: fg_bg_depth

Mask prediction: mask_pred

Depth map prediction: depth_pred

All Images are available in this link https://drive.google.com/drive/folders/1E8V1RUeF--_THXxJ06j7L4vYxOGnlVIk?usp=sharing

**Stats**
using np.std(numpy_images, axis=(0, 2, 3)); np.mean(numpy_images, axis=(0, 2, 3))

    Foreground Background images:
        Mean: [0.582124 0.579472 0.567419]
        Standard Deviation: [0.22423423 0.23696194 0.24423425]

**Decision criteria**

What to build
Architecture
Image size, Batch size, epochs, 
Loss function
handling OOM
What LR, optimiser and scheduler to use.

**What can we do with the dataset **
This became an important problem that required quite a bit of research. 
We could do multiple things - Monocular depth estimation, which is trying to predict the depth of the picture; 
A second one could be image segmentation where the goal is to train a neural network to output a pixel-wise mask of the image. 
This helps in understanding the image at a much lower level, i.e., the pixel level

I tried my hand at both, made a lot of mistakes and then realised that both of them are pretty similar (but divergent) in approach, and the ebst algorithms for each is different

***Image segmentation***
In Image segmentation, I am just trying to have two classes - foreground and background. the foreground is the image that i want to segment (develop the pixel level outline for)
Since I am jsut concerned with the two classes, the type of object (dog/cat) did not matter, only differentiating the foreground and bg did. 
So everything was run in grayscale, and then each iamge was stripped down to a 3 kb image because i couldn/t deal with colab's memory issues anymore.

I used three different kinds of architectures here. My first (and elaborate and very heavy) Resnet 34 architecture, the custom Resnet18 architecture were a work of art:(
but they did not give the requisite model validations. (I had a 70/30 train/validate sample of the 4,00,000 images that were already present)
batch size was an issue, so converted the entire train into batches of 200k. 
I did lost one folder of about 10K images to bad naming conventions and the like, so the final training was on around 3,60,000 images for the segmentation.

![Model](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA15/model.png)

UNet(

(convblock1): Sequential(

(0): Conv2d(6, 64, kernel\_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

(1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(2): ReLU()

(3): Conv2d(64, 64, kernel\_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

(4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(5): ReLU()

)

(pool1): MaxPool2d(kernel\_size=2, stride=2, padding=0, dilation=1, ceil\_mode=False)

(convblock2): Sequential(

(0): Conv2d(64, 128, kernel\_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

(1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(2): ReLU()

(3): Conv2d(128, 128, kernel\_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

(4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(5): ReLU()

)

(pool2): MaxPool2d(kernel\_size=2, stride=2, padding=0, dilation=1, ceil\_mode=False)

(convblock3): Sequential(

(0): Conv2d(128, 256, kernel\_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

(1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(2): ReLU()

(3): Conv2d(256, 256, kernel\_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

(4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(5): ReLU()

)

(pool3): MaxPool2d(kernel\_size=2, stride=2, padding=0, dilation=1, ceil\_mode=False)

(convblock4): Sequential(

(0): Conv2d(256, 512, kernel\_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

(1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(2): ReLU()

(3): Conv2d(512, 512, kernel\_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

(4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(5): ReLU()

)

(pool4): MaxPool2d(kernel\_size=2, stride=2, padding=0, dilation=1, ceil\_mode=False)

(base): Sequential(

(0): Conv2d(512, 1024, kernel\_size=(3, 3), stride=(1, 1), bias=False)

(1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(2): ReLU()

(3): Conv2d(1024, 1024, kernel\_size=(3, 3), stride=(1, 1), bias=False)

(4): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(5): ReLU()

(6): ConvTranspose2d(1024, 512, kernel\_size=(3, 3), stride=(2, 2), padding=(1, 1), output\_padding=(1, 1), bias=False)

)

(convblock5): Sequential(

(0): Conv2d(1024, 512, kernel\_size=(3, 3), stride=(1, 1), bias=False)

(1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(2): ReLU()

(3): Conv2d(512, 512, kernel\_size=(3, 3), stride=(1, 1), bias=False)

(4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(5): ReLU()

(6): ConvTranspose2d(512, 256, kernel\_size=(3, 3), stride=(2, 2), padding=(1, 1), output\_padding=(1, 1), bias=False)

)

(convblock6): Sequential(

(0): Conv2d(512, 256, kernel\_size=(3, 3), stride=(1, 1), bias=False)

(1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(2): ReLU()

(3): Conv2d(256, 256, kernel\_size=(3, 3), stride=(1, 1), bias=False)

(4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(5): ReLU()

(6): ConvTranspose2d(256, 128, kernel\_size=(3, 3), stride=(2, 2), padding=(1, 1), output\_padding=(1, 1), bias=False)

)

(convblock7): Sequential(

(0): Conv2d(256, 128, kernel\_size=(3, 3), stride=(1, 1), bias=False)

(1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(2): ReLU()

(3): Conv2d(128, 128, kernel\_size=(3, 3), stride=(1, 1), bias=False)

(4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(5): ReLU()

(6): ConvTranspose2d(128, 64, kernel\_size=(3, 3), stride=(2, 2), padding=(1, 1), output\_padding=(1, 1), bias=False)

)

(convblock\_final): Sequential(

(0): Conv2d(128, 64, kernel\_size=(3, 3), stride=(1, 1), bias=False)

(1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(2): ReLU()

(3): Conv2d(64, 64, kernel\_size=(3, 3), stride=(1, 1), bias=False)

(4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(5): ReLU()

(6): Conv2d(64, 1, kernel\_size=(3, 3), stride=(1, 1), bias=False)

(7): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track\_running\_stats=True)

(8): ReLU()

)

)

Steps:

Zip file of 200K images of BG, BGFG, BGFG Masks & DepthMaps with subfolders and naming convention, extracted using Python's ZipFile library.
Train in a loop 

 Augmentation across all 4 ImageTypes, Resize,normalise, totensor

The final architecture that worked was the U-NET architecture.


Inputs are of size 192x192, BG was 192*192 and FG 96*96. In the final model, I had outputs are 96x96. 

The U-Net architecture is built upon the Fully Convolutional Network and modified in a way that it yields better segmentation. 
Compared to FCN-8, the two main differences are (1) U-net is symmetric and (2) the skip connections between the downsampling path and the upsampling path apply a concatenation operator instead of a sum. 
Unet follows the architecture conv_layer1 -> conv_layer2 -> max_pooling -> dropout(optional) for contractive path and 
conv_2d_transpose -> concatenate -> conv_layer1 -> conv_layer2 for expansive path. 
Hsd to make  changes to take in_channels as 6 channels in this architecture

Training
training was an impossible achievement, managed by reducing image size, running more epochs in favour of larger batch size, running by making sure last.pt was saved and reused
judiciously letting go of epochs where the training was mostly complete and I had achieved a certain outcome already etc.
trained on lower resolution images first and then the image resolution was increased.

**Loss functions**
- so many were available. 

DICE was best, and then SSIM was easiest to obtain and most stable.
BCEWithLogitLoss seemed better for smaller image sizes but blew up as image res became larger. 

I was not able to finalise on a specific loss, because the loss metrics lept fluctuating, I think SSIM was more stable so that should be reported.

1. SSIM
2. BCEWithLogitLoss
3. Dice Loss
4. RMS Loss
5. Dice Coeff(accuracy)
6. IOU ( Accuracy )
6. Pixelwise Comparison

**LR and scheduler**
Did not expirement crazily with this. 
used Adam optimiser and ReduceLROnPlateau was used as scheduler.
tried with LR on Plateau with patience of 3 and threshold 1e-4, but i lost my patience sooner. ADAM it was finally.



## Model Evaluation
Didnt do a good job on model evaluation. 

##Code Links 

Model : https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA15/unet.py

Colab : https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA15/ImageSeg_using_UNET_MVEVA15.ipynb

Train, test, dataloader , Albumentation, trans : https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS15/Code/utils.py

Dataset : https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS14/Assignment14Images/

Loss functions are in the colabnotebook and graphs are in https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA15/Images/



references

1. https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6

2. https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47

3.https://www.tensorflow.org/tutorials/images/segmentation
