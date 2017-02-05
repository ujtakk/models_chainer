#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

class LeNet(chainer.Chain):
    def __init__(self, image_colors=3, class_labels=10):
        super(LeNet, self).__init__(
            conv0=L.Convolution2D(None, 16, ksize=3, pad=1),
            conv1=L.Convolution2D(None, 32, ksize=3, pad=1),
            full2=L.Linear(None, 256),
            full3=L.Linear(None, class_labels),
        )

    def __call__(self, x, train=True):
        h = self.conv0(x)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.conv1(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.full2(h)
        h = F.relu(h)
        return self.full3(h)
