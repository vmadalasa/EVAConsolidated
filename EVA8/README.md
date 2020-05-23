**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

Colab file:(https://colab.research.google.com/drive/17Gxfnr4UxEpRPvtS4WZJCVLk4njc3_Hm?usp=sharing)

To run the model, we need to upload all the necessary packagesto the colab directory. The packages can be found in the S7 folder of EVA4 repo.


### **Objective**

Achieve the following on the **CIFAR-10** dataset:

- Extract the ResNet18 model from this repository and add it to your API/repo
- Use data loader, train, test, and utils code to train ResNet18 on Cifar10.
- Use default ResNet18 code (so params are fixed).
- achieve **85%** accuracy, as many epochs as required.

### **Model Statistics:**

- ResNet18 - BasicBlock - [2,2,2,2]
- Batch Size: 128
- Number of Parameters: 11,173,962
- Epochs: 50

### **Results**

Achieved accuracy of

**Test - 91.58%**
**Train - 96.66%**

Misclassified Images:

![MissClassifiedImages.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA8/Images/MissClassify.png)

Train and Test Accuracies and Loss:

![Test-Train Accuracy and Loss.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA8/Images/LossandAcc.png)

Train vs Test Accuracy:

![Test-vs-Train Accuracy.png](https://github.com/vmadalasa/EVAConsolidated/blob/master/EVA8/Images/TestvTrainAcc.png)

Class Wise Accuracies:

Accuracy of plane : 93 %

Accuracy of   car : 97 %

Accuracy of  bird : 88 %

Accuracy of   cat : 85 %

Accuracy of  deer : 93 %

Accuracy of   dog : 89 %

Accuracy of  frog : 92 %

Accuracy of horse : 91 %

Accuracy of  ship : 92 %
Accuracy of truck : 93 %
