# EVA4S5
EVA4 S5 assignment

Goal of the assignment is to acheive 99.4% test accuracy in less than 10kparameters for MNIST dataset

First approach was to get the skeleton model straight so that only minor tweaks were needed. 
The first skeleton model tried had 10 channels only per convolution and 3 conv layers in the first block, max pool, transition block , 5 more convolutions - the accuracy of the model was quite low. This had about 6790 parameters only. 

The second model is a second skeleton model where there are 12 channels in most conv layers with the immediate prior layer to max pool and gap having more channels. The model performance for this model seems to be better. The model had 9292 params, within the 10 k limit, and also introduced a gap layer.
The architecture is as given below for the second skeleton model
Input size 28X28X1; output_size = 26X26X12; RF 3X3 - Convinput 
Input size 26X26X12; output_size = 24X24X12; RF 5X5 - convblock1A
Input size 24X24X12; output_size = 22X22X20; RF 7X7 
Max pool to get this to Input size 22X22X20; output_size = 11X11X20; RF 14X14 - pool1 
Input size 11X11X20 output_size = 11X11X12; RF 14X14 - tranblock1

Convblock 2 
Input size 11X11X12; output_size = 9X9X12; RF 16X16 - convblock2A-
Input size 9X9X12; output_size = 7X7X16; RF 18X18 convblock2B
Input size 7X7X16; output_size = 5X5X16; RF 20X20 convblock2C-

We do self.gap layer at kernel size 5 - gap 
finally transblock 1X1 for Input size 1X1X16; output_size = 1X1X10; RF 20X20 - convblockEnd WITHOUT batch norm or dropout here

3rd Model
3rd Model used batchnorm to improve model efficiency. This was used in all layers

4th Model 
4th model used dropout values to reduce overfitting
4th Model, i tried different dropout values. Also tried differential dropout values for conv block 1 and 2 because of reading it in https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/. 
Eventually figured that 0.1 is the best dropout

5th model
used image rotation to train model better by providing rotation

Again. i tried image rotation of 7,7 and 9,9

For 7,7 image rotation
train accurcy is 98.82 but test accuracy is lower at 99.29
For 9X9 - 
train accuracy 98.72; test accuracy 99.33

I don't think these are significantly different, but will go with 9,9 for now.

6th Model
used batch size of 64 and 32. Found that 64 has better results than 128.

Extra
1. tried with LR optimisation using Adam to reduce to 8 k parameters 
2. experimented with maxpool layer being replaced by conv with stride =2 so this can become a learning layer. 


