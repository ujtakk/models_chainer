#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

class LeNet(chainer.Chain):
    def __init__(self, image_colors=3, class_labels=10):
        super(LeNet, self).__init__(
            conv0=L.Convolution2D(None, 16, 3, 1),
            bn0=L.BatchNormalization(16),
            conv1=L.Convolution2D(None, 32, 3, 1),
            bn1=L.BatchNormalization(32),
            full0=L.Linear(None, 256),
            full1=L.Linear(None, class_labels),
        )

    def __call__(self, x, train=True):
        h = self.conv0(x)
        h = self.bn0(h, test=not train)
        h = F.relu(h)
        h = self.conv1(h)
        h = self.bn1(h, test=not train)
        h = F.relu(h)
        h = self.full0(h)
        h = F.relu(h)
        return self.full1(h)
