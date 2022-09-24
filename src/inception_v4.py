from functools import partial

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import XavierUniform

from .utils import load_pretrained
from .registry import register_model
from .layers.conv_norm_act import Conv2dNormActivation
from .layers.pooling import GlobalAvgPooling
from mission.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT

'''
1. 基本结构和timm都是一样的，没有太大差别。
2. 同样将 conv2d+bn+relu 改成了Conv2dNormActivation
3. 初始化权重后期统一整改。
4. 全局平均池化使用提出来公用的GlobalAvgPooling
'''


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'dataset_transform': {
            'transforms_imagenet_train': {
                'image_resize': 224,
                'scale': (0.08, 1.0),
                'ratio': (0.75, 1.333),
                'hflip': 0.5,
                'interpolation': 'bilinear',
                'mean': IMAGENET_DEFAULT_MEAN,
                'std': IMAGENET_DEFAULT_STD,
            },
            'transforms_imagenet_eval': {
                'image_resize': 224,
                'crop_pct': DEFAULT_CROP_PCT,
                'interpolation': 'bilinear',
                'mean': IMAGENET_DEFAULT_MEAN,
                'std': IMAGENET_DEFAULT_STD,
            },
        },
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'inception_v4': _cfg(url='')
}

weight = XavierUniform()
norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.9997)
Conv2dNormActivation = partial(Conv2dNormActivation, norm=norm, weight=weight)


class Stem(nn.Cell):
    """
    Inceptionv4 stem

    """

    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv2d_1a_3x3 = Conv2dNormActivation(in_channels, 32, kernel_size=3, stride=2, pad_mode='valid')
        self.conv2d_2a_3x3 = Conv2dNormActivation(32, 32, kernel_size=3, stride=1, pad_mode='valid')
        self.conv2d_2b_3x3 = Conv2dNormActivation(32, 64, kernel_size=3, stride=1, pad_mode='pad')

        self.mixed_3a_branch_0 = nn.MaxPool2d(3, stride=2)
        self.mixed_3a_branch_1 = Conv2dNormActivation(64, 96, kernel_size=3, stride=2, pad_mode='valid')

        self.mixed_4a_branch_0 = nn.SequentialCell([
            Conv2dNormActivation(160, 64, kernel_size=1, stride=1),
            Conv2dNormActivation(64, 96, kernel_size=3, stride=1, pad_mode='valid')])

        self.mixed_4a_branch_1 = nn.SequentialCell([
            Conv2dNormActivation(160, 64, kernel_size=1, stride=1),
            Conv2dNormActivation(64, 64, kernel_size=(1, 7), stride=1, pad_mode='same'),
            Conv2dNormActivation(64, 64, kernel_size=(7, 1), stride=1, pad_mode='same'),
            Conv2dNormActivation(64, 96, kernel_size=3, stride=1, pad_mode='valid')])

        self.mixed_5a_branch_0 = Conv2dNormActivation(192, 192, kernel_size=3, stride=2, pad_mode='valid')
        self.mixed_5a_branch_1 = nn.MaxPool2d(3, stride=2)

        self.concat = ops.Concat(1)

    def construct(self, x):
        """construct"""
        x = self.conv2d_1a_3x3(x)  # 149 x 149 x 32
        x = self.conv2d_2a_3x3(x)  # 147 x 147 x 32
        x = self.conv2d_2b_3x3(x)  # 147 x 147 x 64

        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = self.concat((x0, x1))  # 73 x 73 x 160

        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x = self.concat((x0, x1))  # 71 x 71 x 192

        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = self.concat((x0, x1))  # 35 x 35 x 384
        return x


class InceptionA(nn.Cell):
    """InceptionA"""

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch_0 = Conv2dNormActivation(384, 96, kernel_size=1, stride=1)
        self.branch_1 = nn.SequentialCell([
            Conv2dNormActivation(384, 64, kernel_size=1, stride=1),
            Conv2dNormActivation(64, 96, kernel_size=3, stride=1, pad_mode='pad')])

        self.branch_2 = nn.SequentialCell([
            Conv2dNormActivation(384, 64, kernel_size=1, stride=1),
            Conv2dNormActivation(64, 96, kernel_size=3, stride=1, pad_mode='pad'),
            Conv2dNormActivation(96, 96, kernel_size=3, stride=1, pad_mode='pad')])

        self.branch_3 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same'),
            Conv2dNormActivation(384, 96, kernel_size=1, stride=1)])

        self.concat = ops.Concat(1)

    def construct(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.concat((x0, x1, x2, x3))
        return x4


class InceptionB(nn.Cell):
    """InceptionB"""

    def __init__(self):
        super(InceptionB, self).__init__()
        self.branch_0 = Conv2dNormActivation(1024, 384, kernel_size=1, stride=1)
        self.branch_1 = nn.SequentialCell([
            Conv2dNormActivation(1024, 192, kernel_size=1, stride=1),
            Conv2dNormActivation(192, 224, kernel_size=(1, 7), stride=1, pad_mode='same'),
            Conv2dNormActivation(224, 256, kernel_size=(7, 1), stride=1, pad_mode='same'),
        ])
        self.branch_2 = nn.SequentialCell([
            Conv2dNormActivation(1024, 192, 1, kernel_size=1, stride=1),
            Conv2dNormActivation(192, 192, kernel_size=(7, 1), stride=1, pad_mode='same'),
            Conv2dNormActivation(192, 224, kernel_size=(1, 7), stride=1, pad_mode='same'),
            Conv2dNormActivation(224, 224, kernel_size=(7, 1), stride=1, pad_mode='same'),
            Conv2dNormActivation(224, 256, kernel_size=(1, 7), stride=1, pad_mode='same')
        ])
        self.branch_3 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same'),
            Conv2dNormActivation(1024, 128, kernel_size=1, stride=1)
        ])
        self.concat = ops.Concat(1)

    def construct(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.concat((x0, x1, x2, x3))
        return x4


class ReductionA(nn.Cell):
    """ReductionA"""

    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch_0 = Conv2dNormActivation(384, 384, kernel_size=3, stride=2, pad_mode='valid')
        self.branch_1 = nn.SequentialCell([
            Conv2dNormActivation(384, 192, kernel_size=1, stride=1),
            Conv2dNormActivation(192, 224, kernel_size=3, stride=1, pad_mode='pad'),
            Conv2dNormActivation(224, 256, kernel_size=3, stride=2, pad_mode='valid'),
        ])
        self.branch_2 = nn.MaxPool2d(3, stride=2)
        self.concat = ops.Concat(1)

    def construct(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.concat((x0, x1, x2))
        return x3


class ReductionB(nn.Cell):
    """ReductionB"""

    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch_0 = nn.SequentialCell([
            Conv2dNormActivation(1024, 192, kernel_size=1, stride=1),
            Conv2dNormActivation(192, 192, kernel_size=3, stride=2, pad_mode='valid'),
        ])
        self.branch_1 = nn.SequentialCell([
            Conv2dNormActivation(1024, 256, kernel_size=1, stride=1),
            Conv2dNormActivation(256, 256, kernel_size=(1, 7), stride=1, pad_mode='same'),
            Conv2dNormActivation(256, 320, kernel_size=(7, 1), stride=1, pad_mode='same'),
            Conv2dNormActivation(320, 320, kernel_size=3, stride=2, pad_mode='valid')
        ])
        self.branch_2 = nn.MaxPool2d(3, stride=2)
        self.concat = ops.Concat(1)

    def construct(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.concat((x0, x1, x2))
        return x3  # 8 x 8 x 1536


class InceptionC(nn.Cell):
    """InceptionC"""

    def __init__(self):
        super(InceptionC, self).__init__()
        self.branch_0 = Conv2dNormActivation(1536, 256, kernel_size=1, stride=1)

        self.branch_1 = Conv2dNormActivation(1536, 384, kernel_size=1, stride=1)
        self.branch_1_1 = Conv2dNormActivation(384, 256, kernel_size=(1, 3), stride=1, pad_mode='same')
        self.branch_1_2 = Conv2dNormActivation(384, 256, kernel_size=(3, 1), stride=1, pad_mode='same')

        self.branch_2 = nn.SequentialCell([
            Conv2dNormActivation(1536, 384, kernel_size=1, stride=1),
            Conv2dNormActivation(384, 448, kernel_size=(3, 1), stride=1, pad_mode='same'),
            Conv2dNormActivation(448, 512, kernel_size=(1, 3), stride=1, pad_mode='same'),
        ])
        self.branch_2_1 = Conv2dNormActivation(512, 256, kernel_size=(1, 3), stride=1, pad_mode='same')
        self.branch_2_2 = Conv2dNormActivation(512, 256, kernel_size=(3, 1), stride=1, pad_mode='same')

        self.branch_3 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same'),
            Conv2dNormActivation(1536, 256, kernel_size=1, stride=1)
        ])

        self.concat = ops.Concat(1)

    def construct(self, x):
        """construct"""
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x1_1 = self.branch_1_1(x1)
        x1_2 = self.branch_1_2(x1)
        x1 = self.concat((x1_1, x1_2))
        x2 = self.branch_2(x)
        x2_1 = self.branch_2_1(x2)
        x2_2 = self.branch_2_2(x2)
        x2 = self.concat((x2_1, x2_2))
        x3 = self.branch_3(x)
        return self.concat((x0, x1, x2, x3))


class InceptionV4(nn.Cell):
    """
    Inceptionv4 architecture
    """

    def __init__(self, in_channels=3, num_classes=1000, drop_rate=0.2):
        super(InceptionV4, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for _ in range(4):
            blocks.append(InceptionA())
        blocks.append(ReductionA())
        for _ in range(7):
            blocks.append(InceptionB())
        blocks.append(ReductionB())
        for _ in range(3):
            blocks.append(InceptionC())
        self.features = nn.SequentialCell(blocks)

        self.pool = GlobalAvgPooling()
        self.dropout = nn.Dropout(1 - drop_rate)
        self.num_features = 1536
        self.classifier = nn.Dense(self.num_features, num_classes)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes):
        self.classifier = nn.Dense(self.num_features, num_classes)

    def get_features(self, x):
        self.features(x)
        return x

    def construct(self, x):
        x = self.get_features(x)
        x = self.pool(x)
        x = self.classifier(x)


@register_model
def inception_v4(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['inception_v4']
    model = InceptionV4(num_classes=num_classes, in_channels=in_channels, **kwargs)
    model.dataset_transform = default_cfg['dataset_transform']

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model

