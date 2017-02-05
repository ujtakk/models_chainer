#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import chainer

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        c, h, w = image.shape

        if c == 1:
            image = np.repeat(image, 3, 0)

        if c == 4:
            print(self.base._pairs[i])

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image  = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label

def get_imagenet():
    # Customizable parameters
    path_root  = "/home/work/takau/imagenet_data/ImageNet_2012/"
    file_mean  = "/home/work/takau/models_chainer/datasets/imagenet_mean.npy"
    file_train = "/home/work/takau/models_chainer/datasets/imagenet_train.txt"
    file_val   = "/home/work/takau/models_chainer/datasets/imagenet_val.txt"
    insize     = 224

    mean = np.load(file_mean)

    train = PreprocessedDataset(file_train, path_root+'train', mean, insize)
    val   = PreprocessedDataset(file_val, path_root+'val', mean, insize, False)

    return train, val
