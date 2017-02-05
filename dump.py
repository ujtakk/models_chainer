#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import exists, join
import numpy as np

import make

def dump(name, filelist, xy_list, num_labels):
    for num in range(num_labels):
        if not exists(join(name, str(num))):
            os.makedirs(join(name, str(num)))

    lut = np.zeros(num_labels, dtype=np.int32)
    with open(filelist, "w") as fl:
        for image, label in xy_list:
            filename = join(name, str(label), "img{}.dat".format(lut[label]))
            fl.write("{} {}\n".format(filename, label))
            with open(filename, "wb") as f:
                np.savetxt(f, image.flatten(), fmt="%8.8f")
            lut[label] += 1

def optparse():
    import argparse
    parser = argparse.ArgumentParser(description='Dump script with chainer')

    parser.add_argument('--dataset', '-d', default='mnist',
                        help='The dataset to use: mnist, cifar10 or cifar100')

    return parser.parse_args()

def main():
    args = optparse()

    # Load the dataset
    dataset = make.dataset(args.dataset)
    image_colors, class_labels = dataset['attr']
    train, test = dataset['data']

    dump("{}/train".format(args.dataset),
         "{}_train.txt".format(args.dataset),
         train, class_labels)

    dump("{}/test".format(args.dataset),
         "{}_test.txt".format(args.dataset),
         test, class_labels)

if __name__ == '__main__':
    main()
