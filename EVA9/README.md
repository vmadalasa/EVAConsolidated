Readme.md
**Assignment 9 File**

**Members:**

Jahnavi Ramagiri

Sachin Sharma

Madalasa Venkataraman

Syed Abdul Khader

**Colab File Link**

[https://colab.research.google.com/drive/1M3Uty8zYc1Iom1nSw\_A57sI7fMWiC3Mo](https://colab.research.google.com/drive/1M3Uty8zYc1Iom1nSw_A57sI7fMWiC3Mo)

**Objective**

1. Move your last code&#39;s transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
2. Please make sure that your test\_transforms are simple and only using ToTensor and Normalize
3. Implement GradCam function as a module.
4. Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
5. Target Accuracy is 87%

Once Done:

1. Paste your Albumentation File Code
2. Paste your GradCam Module&#39;s code
3. Paste your notebook file code
4. What is your final Validation Accuracy?
5. Share the link to your Repo.

**Our Network**

Our custom Resnet18 is modified to pass a 8X8 image to GradCAM

**Packages**

Are available in the package folder

- used modules for data.py, train.py, test.py, summary.py
- Albumentation is in Albtransform.py
- Resnet DNN modified is in CusResNet.py
- Grad\_model.py and plot.py define the GradCAM and plot the GradCAM respectively

Model Statistics

ResNet18 – Custom Resnet to provide 8X8 image to GradCAM

Batch Size: 128

Number of Parameters: 11,173,962

Epochs: 15

Achieved accuracy of

**Test - 86.95%**   **Train – 90.65%**

**Images**

Misclassified images

![MisClassifiedImages.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA9/Images/MisClassifiedImages.png)

GradCAM images

![GradCAMImages.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA9/Images/GradCAM.png)

Heatmap

![HeatmapImages.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA9/Images/Heatmap.png)

Train and Test Accuracies and Loss:

![Loss.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA9/Images/Loss.png)

Train vs Test Accuracy:

![Accuracy.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA9/Images/Accuracy.png)

Class Wise Accuracy

Accuracy of plane : 85 %

Accuracy of car : 97 %

Accuracy of bird : 85 %

Accuracy of cat : 79 %

Accuracy of deer : 92 %

Accuracy of dog : 80 %

Accuracy of frog : 83 %

Accuracy of horse : 86 %

Accuracy of ship : 88 %

Accuracy of truck : 91 %
