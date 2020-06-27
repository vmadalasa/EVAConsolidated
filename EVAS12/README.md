**Assignment 12 File**

**Members:**

Madalasa Venkataraman

Syed Abdul Khader

Jahnavi Ramagiri

Sachin Sharma

**Assignment A**

Objectives

1. Download this [TINY IMAGENET (Links to an external site.)](http://cs231n.stanford.edu/tiny-imagenet-200.zip) dataset.
2. Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy.
3. Submit Results. Of course, you are using your own package for everything. You can look at [this (Links to an external site.)](https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/Train_ResNet_On_Tiny_ImageNet.ipynb) for reference.

**Colab File Link**

[https://colab.research.google.com/drive/1jdc12gHub3sfrzu73YDFM-7-S4ddwpnX](https://colab.research.google.com/drive/1jdc12gHub3sfrzu73YDFM-7-S4ddwpnX)

Assignment B

  1. Download 50 images of dogs.
  2. Use [this (Links to an external site.)](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) to annotate bounding boxes around the dogs.
  3. Download JSON file.
  4. Describe the contents of this JSON file in FULL details (you don&#39;t need to describe all 10 instances, anyone would work).
  5. Refer to this [tutorial (Links to an external site.)](https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203). Find out the best total numbers of clusters. Upload link to your Colab File uploaded to GitHub.

**Colab File Link**

[https://colab.research.google.com/drive/18ALq8\_6NpbI3ghGYSUuEyAkzKav28M0m](https://colab.research.google.com/drive/18ALq8_6NpbI3ghGYSUuEyAkzKav28M0m)

**Our Network**

Resnet18

**Packages**

Are available in the package folder

- Data, test and train packages.
- Imagenet\_dataloader package that extracts classes, does the shuffle and assignment to train and val subsets
- Resnet DNN

**Model Statistics**

ResNet18

Batch Size: 256

Number of Parameters: 11,271,432

Epochs: 20

Achieved accuracy of 53.58%

For Experiment B

Obtained pictures of 50 dogs from the internet.

Used [http://www.robots.ox.ac.uk/~vgg/software/](http://www.robots.ox.ac.uk/~vgg/software/) to add images from local, annotate images using bounding boxes

JSON has details on:

name of object,

object\_type,

imagewithinpicture (to identify whether dog is completely within the picture or extends beyond it

objectname - dogsleeping, dog standing, doginprofile, doglying, dogsitting, dogwithleash

image\_quality (good illumination, blurred, object in frontal view)

Based on this, identified the size of the bounding bo relative to the size of the image, height ratio and width ratio, log(height ratio), log(width ratio), relative positions etc.

Used the above data to run a k-means cluster and ran elbow plot. 3 distinct clusters could be identified
