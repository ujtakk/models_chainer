#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

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

    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__(
            conv0=L.Convolution2D(None, out_channels, ksize, pad=pad),
            bn0=L.BatchNormalization(out_channels),
            conv1=L.Convolution2D(None, out_channels, ksize, pad=pad),
            bn1=L.BatchNormalization(out_channels),
            conv2=L.Convolution2D(None, out_channels, ksize, pad=pad),
            bn2=L.BatchNormalization(out_channels),
        )

    def __call__(self, x, train=True):
        h = self.conv0(x)
        h = self.bn0(h, test=not train)
        h = F.relu(h)
        h = self.conv1(h)
        h = self.bn1(h, test=not train)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn2(h, test=not train)
        h = F.relu(h)
        return h


class LeBlock(chainer.Chain):

    def __init__(self, image_colors=3, class_labels=10):
        super(LeBlock, self).__init__(
            block=Block(32, image_colors),
            full0=L.Linear(None, 256),
            full1=L.Linear(None, class_labels),
        )
        self.train = True

    def __call__(self, x):
        h = self.block(x, self.train)
        h = self.full0(h)
        h = F.relu(h)
        return self.full1(h)
