## ResNet V1

This is a Keras implementation of the residual network presented in [this article](https://arxiv.org/abs/1512.03385). The model is trained on the CIFAR-10 dataset for 200 epochs with a decaying learning rate. It is a general implementation that can be used to generate bigger models, but does not use the initial paper's exact blocks structure.

My goal with this implementation was to get some experience working first-hand with a residual network, as well as to get a sense of the model's performance.

Further comparison can be found on the ResNet V2 part of this repository.

| Model        | n  | Accuracy(%) | Loss  |
| -------------|:---:| -----:| --- |
| ResNet20 V1  |  3  | XX,XX |     |
| ResNet56 V1  |  9  | 91,04 | 0,4974 |
| ResNet110 V1 |  18 | 91,51 | 0,4736 |
