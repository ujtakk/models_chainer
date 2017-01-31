#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import chainer
import chainer.functions as F
import chainer.links as L

from ._resnet import BottleNeckA, BottleNeckB, Block

class ResNet101(chainer.Chain):

    insize = 224

    def __init__(self, image_colors=3, class_labels=10):
        w = math.sqrt(2)
        super(ResNet101, self).__init__(
            conv1=L.Convolution2D(image_colors, 64, 7, 2, 3, w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block(3, 64, 64, 256, 1),
            res3=Block(4, 256, 128, 512),
            res4=Block(23, 512, 256, 1024),
            res5=Block(3, 1024, 512, 2048),
            fc=L.Linear(2048, class_labels),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x):
        self.clear()
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        # h = F.average_pooling_2d(h, 7, stride=1)
        h = F.average_pooling_2d(h, 1, stride=1)

        return self.fc(h)

