#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

# from ._ResNet import BottleNeckA, BottleNeckB, Block

class Block(chainer.Chain):

    """A convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, channels, ksize=3, pad=1):
        super(Block, self).__init__(
            bn0=L.BatchNormalization(channels),
            conv0=L.Convolution2D(None, channels, ksize, pad=pad),
            bn1=L.BatchNormalization(channels),
            conv1=L.Convolution2D(None, channels, ksize, pad=pad),
        )

    def __call__(self, x, train=True):
        h = self.bn0(x, test=not train)
        h = F.relu(h)
        h = self.conv0(h)
        h = self.bn1(h, test=not train)
        h = F.relu(h)
        h = self.conv1(h)
        return x + h

class ResBlock(chainer.Chain):

    def __init__(self, image_colors=3, class_labels=10):
        super(ResBlock, self).__init__(
            conv=L.Convolution2D(image_colors, 32, ksize=3, pad=1),
            block=Block(32),
            fc0=L.Linear(None, 256),
            fc1=L.Linear(None, class_labels),
        )
        self.train = True

    def __call__(self, x):
        h = self.conv(x)
        h = self.block(h, self.train)
        h = self.fc0(h)
        h = F.relu(h)
        return self.fc1(h)
