#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer
import make

def optparse():
    import argparse

    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')

    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: mnist, cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--arch', '-a', default='vgg',
                        help='Convnet architecture')

    return parser.parse_args()

def main():
    args = optparse()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    dataset = make.dataset(args.dataset)
    image_colors, class_labels = dataset['attr']
    train, test = dataset['data']

    model = make.model(args.arch, image_colors, class_labels)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Run the training
    optim   = chainer.optimizers.AdaGrad().setup(model)
    trainer = make.trainer(args, model, optim, train, test)
    trainer.run()

if __name__ == '__main__':
    main()
