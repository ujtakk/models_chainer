#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

class MLP(chainer.Chain):
    def __init__(self, image_colors=1, class_labels=10):
        super().__init__()
        with self.init_scope():
            self.full1 = L.Linear(None, 1000)
            self.full2 = L.Linear(None, class_labels)

    def __call__(self, x, train=True):
        h = self.full1(x)
        h = F.relu(h)
        return self.full2(h)
