# Inception_V4

## InceptionV4 Description

Inception-v4 is a convolutional neural network architecture that builds on previous iterations of the Inception family by simplifying the architecture and using more inception modules than Inception-v3. This idea was proposed in the paper Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, published in 2016.

[Paper](https://arxiv.org/pdf/1602.07261.pdf) Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi. Computer Vision and Pattern Recognition[J]. 2016.

## Model architecture
The overall network architecture of InceptionV4 is show below:

[Link](https://arxiv.org/pdf/1602.07261.pdf)

## Training process

```shell
  export CUDA_VISIBLE_DEVICES=0
  python train.py --model googlenet --data_url ./dataset/imagenet
```

```text
epoch: 1 step: 1251, loss is 6.49775
Epoch time: 1487493.604, per step time: 1189.044
epoch: 2 step: 1251, loss is 5.6884665
Epoch time: 1421838.433, per step time: 1136.561
epoch: 3 step: 1251, loss is 5.5168786
Epoch time: 1423009.501, per step time: 1137.498
```

## [Eval process](#contents)

### Usage

```shell
python validate.py --model googlenet --data_url ./dataset/imagenet --checkpoint_path=[CHECKPOINT_PATH]
```

```text
metric: {'Loss': 0.8144, 'Top1-Acc': 0.8009, 'Top5-Acc': 0.9457}
```
