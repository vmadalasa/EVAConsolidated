Readme

**Assignment 10 File**

**Members:**

Madalasa Venkataraman

Syed Abdul Khader

Jahnavi Ramagiri

Sachin Sharma

**Colab File Link**

[https://colab.research.google.com/drive/1tPnSVYYDogj2tv3D9dDWO5P96DZB5OWL](https://colab.research.google.com/drive/1tPnSVYYDogj2tv3D9dDWO5P96DZB5OWL)

**Objective**

1. Pick your last code
2. Make sure  to Add CutOut to your code. It should come from your transformations (albumentations)
3. Use this repo: [https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.)](https://github.com/davidtvs/pytorch-lr-finder)
  1. Move LR Finder code to your modules
  2. Implement LR Finder (for SGD, not for ADAM)
  3. Implement ReduceLROnPlatea: [https://pytorch.org/docs/stable/optim.html#torch.optim.lr\_scheduler.ReduceLROnPlateau (Links to an external site.)](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
4. Find best LR to train your model
5. Use SDG with Momentum
6. Train for 50 Epochs.
7. Show Training and Test Accuracy curves
8. Target 88% Accuracy.
9. Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
10. Submit

Once Done:

1. Paste your S10 Assignment&#39;s GitHub Link - 500PTS
2. Paste the link or upload Training and Test Curves (there should only be 1 graph)- 100PTS
3. What is the test accuracy of your model? - 150PTS (If you have mentioned training accuracies, please comment on your assignment what is test)
4. Share the link or upload an image of 25 misclassified images with GradCam results on top of them- 250PTS

**Our Network**

Our custom Resnet18 is modified to pass a 8X8 image to GradCAM

**Packages**

Are available in the package folder

- used modules for data.py, train.py, test.py, summary.py
- Albumentation is in augmentation.py
- Resnet DNN modified is in CusResNet.py
- py and plot.py define the GradCAM and plot the GradCAM respectively
- We have added the LR\_Finder module in the find\_lr.py package

**Model Statistics**

ResNet18 – Custom Resnet to provide 8X8 image to GradCAM

Batch Size: 128

Number of Parameters: 11,173,962

Epochs: 50

Achieved accuracy of

**Test – 92.42%**   **Train – 97.42%**

**Images**

Misclassified images

![MisClass.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS10/Images/MisClass.png)

GradCAM of misclassified images

Objective - Wrt predicted class (wrong class) to see what went wrong/what was done in the DNN

![GradCAM.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS10/Images/GradCAM.png)

Gradcam images wrt actual (correct) class

Objective - to see what should have been done

![GradCAM2.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS10/Images/GradCAM2.png)

Train and Test Accuracies and Loss:

![Loss.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS10/Images/Loss.png)

Train vs Test Accuracy:

![Accuracy.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVAS10/Images/Accuracy.png)

**Class Wise Accuracy**

Accuracy of plane : 92 %

Accuracy of car : 96 %

Accuracy of bird : 88 %

Accuracy of cat : 84 %

Accuracy of deer : 93 %

Accuracy of dog : 87 %

Accuracy of frog : 95 %

Accuracy of horse : 95 %

Accuracy of ship : 96 %

Accuracy of truck : 95 %
