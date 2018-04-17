#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

class LeNet3x3(chainer.Chain):
    def __init__(self, image_colors=3, class_labels=10):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, ksize=3, pad=1)
            self.conv2 = L.Convolution2D(None, 32, ksize=3, pad=1)
            self.full3 = L.Linear(None, 256)
            self.full4 = L.Linear(None, class_labels)

    def __call__(self, x, train=True):
        h = self.conv1(x)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.full3(h)
        h = F.relu(h)
        return self.full4(h)

class LeNet5x5(chainer.Chain):
    def __init__(self, image_colors=3, class_labels=10):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, ksize=5)
            self.conv2 = L.Convolution2D(None, 32, ksize=5)
            self.full3 = L.Linear(None, 256)
            self.full4 = L.Linear(None, class_labels)

    def __call__(self, x, train=True):
        h = self.conv1(x)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.full3(h)
        h = F.relu(h)
        return self.full4(h)
