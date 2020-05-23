Members: Syed Abdul Khader, Jahnavi Ramagiri, Madalasa Venkataraman, Sachin CV



Colab file:[https://colab.research.google.com/drive/1GOzQzdFFGgejLamHnWxlHmKdGh26-ZnF#scrollTo=ur2bt7oBauPc)

Packages have been created for the following
Data loader; 
Training; 
Test;
Misclassified images;
Model (conv layers);
summary


USED 

DILATION CONV LAYER;
DEPTHWISE SEPARABLE conv layer


Success criteria for the S7 assignment as follows:

Achieve the following on the CIFAR-10 dataset:

The code must utilize GPU

The architecture must be C1C2C3C40 (basically 3 MPs).

Total RF must be more than 44.

Atleast one of the layers must use Depthwise Separable Convolution

Atleast one of the layers must use Dilated Convolution

use GAP (compulsory):- add FC after GAP to target no. of classes (optional)

achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M.



Our Model Statistics:

Number of Parameters: 256,440

reduce overfitting with : Dropout: 0.1

Regularisation: L2, weight decay 1e-5

Normalisation : GhostBatchNormalization

Epochs: 50



Results
Achieved accuracy of

Test - 84.98% Train - 86.66%


Train vs Test Accuracy:


Class Wise Accuracies:

Accuracy of plane : 84 %

Accuracy of car : 92 %

Accuracy of bird : 77 %

Accuracy of cat : 67 %

Accuracy of deer : 84 %

Accuracy of dog : 80 %

Accuracy of frog : 91 %

Accuracy of horse : 86 %

Accuracy of ship : 92 %

Accuracy of truck : 92 %
