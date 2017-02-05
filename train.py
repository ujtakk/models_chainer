#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer
import make

def optparse():
    import argparse

    parser = argparse.ArgumentParser(description='Training script with chainer')

    parser.add_argument('--dataset', '-d', default='mnist',
                        help='The dataset to use: mnist, cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--arch', '-a', default='lenet',
                        help='ConvNet architecture')
    parser.add_argument('--loaderjob', '-j', type=int, default=8,
                        help='Number of parallel data loading processes')
    parser.add_argument('--val_batchsize', '-v', type=int, default=250,
                        help='Validation minibatch size')

    return parser.parse_args()

def main():
    args = optparse()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Load the dataset
    dataset = make.dataset(args.dataset)
    image_colors, class_labels = dataset['attr']
    train, test = dataset['data']

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = make.model(args.arch, image_colors, class_labels)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    # Make the GPU current
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Run the training
    optim   = chainer.optimizers.AdaGrad()
    optim.setup(model)
    trainer = make.trainer(args, model, optim, train, test)
    trainer.run()

    # for l in model.predictor.namedparams():
    #   print(l)
    # print(model.predictor.__class__.__name__)
    make.weight(model)

if __name__ == '__main__':
    main()
