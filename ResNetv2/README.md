## ResNet V2

This is a Keras implementation of the residual network v2 presented in [this article](https://arxiv.org/abs/1603.05027). The model is trained on the CIFAR-10 dataset for 200 epochs. It is a general implementation that can be used to generate bigger models, but does not use the initial paper's exact blocks structure.

My goal with this 2nd implementation was to mess around with the 2nd version of the ResNet, as well as to see the effect of some hyperparameters like learning rate, and whether or not to use data augmentation.

## Results

The results obtained are presented in the table below. More details can be found in the following section.

| Model        | n  | Accuracy(%) | Loss  |
| -------------|:---:| -----:| --- |
| ResNet20 V2  |  2  | 83,35 |   0,9811  |
| ResNet56 V2  |  6  | 86,27 | 0,8192 |
| ResNet110 V2 |  12 | 85,83 | 0,8791 |
| ResNet20 V2 Data augmented  |  2  | 90,01 |  0,5038   |
| ResNet56 V2 Data augmented |  6 | 90,55 | 0,4747 |
| ResNet110 V2 Data augmented|  12 | 91,72 | 0,4461 |
| ResNet20 V2 Data augmented with LR schedule |  2  |91,21 |  0,4328   |
| ResNet56 V2 Data augmented with LR schedule |  6  | 92,92| 0,3948 |
| ResNet110 V2 Data augmented with LR schedule|  12 | 92,96 | 0,3944 |

Compared to the ResNetV1

| Model        | n  | Accuracy(%) | Loss  |
| -------------|:---:| -----:| --- |
| ResNet20 V1  |  3  | XX,XX |     |
| ResNet56 V1  |  9  | 91,04 | 0,4974 |
| ResNet110 V1 |  18 | 91,51 | 0,4736 |

Interestingly, the results with ResNetv2 are only marginally better than with ResNetv1. 

## Methodology

### Baseline

I started by running the baseline algorithms with a basic decaying learning rate and without data augmentation. For the sake of simplicity, I only trained the ResNet20V2, ResNet56V2 and ResNet110V2 versions of the network.

ResNet20V2 Accuracy            |  ResNet20V2 Loss
:-------------------------:|:-------------------------:
![](20v2_acc)  |  ![](20v2_loss)

ResNet56V2 Accuracy            |  ResNet56V2 Loss
:-------------------------:|:-------------------------:
![](56v2_acc)  |  ![](56v2_loss)

ResNet110V2 Accuracy            |  ResNet110V2 Loss
:-------------------------:|:-------------------------:
![](110v2_acc)  |  ![](110v2_loss)

As can be seen in the figures and above, more layers doesn't necessarily mean better results. In fact, at 200 epochs, the ResNet56 version provided a better accuracy than the ResNet110 version.

### With data augmentation

Then, I added data augmentation to the training. This consisted of modifying a bit the training images in order to obtain a bigger dataset (for example, rotating an image, slightly changing the RGB channel values, etc.)

ResNet20V2 with data augmentation Accuracy            |  ResNet20V2 with data augmentation Loss
:-------------------------:|:-------------------------:
![](20v2_data_acc)  |  ![](20v2_data_loss)

ResNet56V2 with data augmentation  Accuracy            |  ResNet56V2 with data augmentation Loss
:-------------------------:|:-------------------------:
![](56v2_data_acc)  |  ![](56v2_data_loss)

ResNet110V2 with data augmentation  Accuracy            |  ResNet110V2 with data augmentation Loss
:-------------------------:|:-------------------------:
![](110v2_data_acc)  |  ![](110v2_data_loss)

The data augmentation gave significantly better results. It gave an improvement of up to 7% on some versions of the residual network, while also pushing the ResNet110's performance over the ResNet56's.


### With custom learning rate schedule

Following the data augmentation, I re-trained the models using a custom learning rate schedule. It starts bigger, and gradually decreases as the epochs advance.

ResNet20V2 with custom LR schedule Accuracy            |  ResNet20V2 with custom LR schedule Loss
:-------------------------:|:-------------------------:
![](20v2_data_LR_acc)  |  ![](20v2_data_LR_loss)

ResNet56V2 with custom LR schedule Accuracy            |  ResNet56V2 with custom LR schedule Loss
:-------------------------:|:-------------------------:
![](56v2_data_LR_acc)  |  ![](56v2_data_LR_loss)

ResNet110V2 with custom LR schedule Accuracy            |  ResNet110V2 with custom LR schedule Loss
:-------------------------:|:-------------------------:
![](110v2_data_LR_acc)  |  ![](110v2_data_LR_loss)

Again, the custom learning rate schedule gave a noticeable improvement, allowing the ResNet110V2's performance to reach 93%. However, due to the increase in training time compared to the ResNet56V2, it does not feel like a big enough improvement.

### Next work

It would be interesting to see how the model fares on another dataset, such as MNIST. It would also be interesting to see if the original paper's suggested block sizes actually offer a better performance than what was achieved in this project.

[20v2_acc]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet20v2_200epoch/Resnet20v2_200epoch_accuracy.png "ResNet20V2_accuracy"
[20v2_loss]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet20v2_200epoch/Resnet20v2_200epochloss.png "ResNet20V2_loss"
[56v2_acc]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet56v2_200epoch/Resnet56v2_200epoch_accuracy.png "ResNet56V2_accuracy"
[56v2_loss]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet56v2_200epoch/Resnet56v2_200epochloss.png "ResNet56V2_loss"
[110v2_acc]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet110v2_200epoch/Resnet110v2_200epoch_accuracy.png "ResNet110V2_accuracy"
[110v2_loss]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet110v2_200epoch/Resnet110v2_200epochloss.png "ResNet110V2_loss"

[20v2_data_acc]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet20v2_200epoch_DataAugmented/Resnet20v2_200epoch_DataAugmented_accuracy.png "ResNet20V2_accuracy_DataAugmented"
[20v2_data_loss]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet20v2_200epoch_DataAugmented/Resnet20v2_200epoch_DataAugmentedloss.png "ResNet20V2_loss_DataAugmented"
[56v2_data_acc]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet56v2_200epoch_DataAugmented/Resnet56v2_200epoch_DataAugmented_accuracy.png "ResNet56V2_accuracy_DataAugmented"
[56v2_data_loss]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet56v2_200epoch_DataAugmented/Resnet56v2_200epoch_DataAugmentedloss.png "ResNet56V2_loss_DataAugmented"
[110v2_data_acc]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet110v2_200epoch_DataAugmented/Resnet110v2_200epoch_DataAugmented_accuracy.png "ResNet110V2_accuracy_DataAugmented"
[110v2_data_loss]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet110v2_200epoch_DataAugmented/Resnet110v2_200epoch_DataAugmentedloss.png "ResNet110V2_loss_DataAugmented"

[20v2_data_LR_acc]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet20v2_200epoch_LRschedule/Resnet20v2_200epoch_LRschedule_accuracy.png "ResNet20V2_accuracy_DataAugmented_LRschedule"
[20v2_data_LR_loss]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet20v2_200epoch_LRscheduleResnet20v2_200epoch_LRscheduleloss.png "ResNet20V2_loss_DataAugmented_LRschedule"
[56v2_data_LR_acc]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet56v2_200epoch_LRschedule/Resnet56v2_200epoch_LRschedule_accuracy.png "ResNet56V2_accuracy_DataAugmented_LRschedule"
[56v2_data_LR_loss]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet56v2_200epoch_LRschedule/Resnet56v2_200epoch_LRscheduleloss.png "ResNet56V2_loss_DataAugmented_LRschedule"
[110v2_data_LR_acc]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet110v2_200epoch_LRschedule/Resnet110v2_200epoch_LRschedule_accuracy.png "ResNet110V2_accuracy_DataAugmented_LRschedule"
[110v2_data_LR_loss]: https://github.com/gadese/ComputerVision_Implementations/tree/develop/ResNetv2/Resnet110v2_200epoch_LRschedule/Resnet110v2_200epoch_LRscheduleloss.png "ResNet110V2_loss_DataAugmented_LRschedule"



