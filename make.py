#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer

def dataset(name):
    from chainer.datasets import get_mnist, get_cifar10, get_cifar100
    from datasets import get_imagenet

    def_attr = lambda image_colors, class_labels: \
        (image_colors, class_labels)

    sets = {
        "mnist": {
            "attr": def_attr(1, 10),
            "data": lambda: get_mnist(ndim=3)
        },
        "cifar10": {
            "attr": def_attr(3, 10),
            "data": lambda: get_cifar10()
        },
        "cifar100": {
            "attr": def_attr(3, 100),
            "data": lambda: get_cifar100()
        },
        "imagenet": {
            "attr": def_attr(3, 1000),
            "data": lambda: get_imagenet()
        }
    }

    print('using {} dataset.'.format(name))

    if name in sets:
        return sets[name]
    else:
        raise RuntimeError('Invalid dataset choice.')

def model(name, image_colors, class_labels):
    import chainer.links as L
    import models as M

    archs = {
        'nin': M.NIN,
        'vgg': M.VGG,
        'mlp': M.MLP,
        'lenet3x3': M.LeNet3x3,
        'lenet5x5': M.LeNet5x5,
        'leblock': M.LeBlock,
        'resblock': M.ResBlock,
        'resnet50': M.ResNet50,
        'resnet101': M.ResNet101,
        'resnet152': M.ResNet152,
    }

    if name in archs:
        return L.Classifier(archs[name](image_colors, class_labels))
    else:
        raise RuntimeError('Invalid architecture choice.')

def trainer(args, model, optimizer, train, test):
    from chainer import training
    from chainer.training import extensions

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    test_iter= chainer.iterators.MultiprocessIterator(
        test, args.val_batchsize, repeat=False, n_processes=args.loaderjob)
    # test_iter  = chainer.iterators.SerialIterator(test, args.batchsize,
    #                                               repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    return trainer

def weight(model):
    import os
    from os.path import exists, join
    import save

    save_func = {
        "Convolution2D":
            lambda name, layer: save.conv(name, layer),
        "Linear":
            lambda name, layer: save.full(name, layer),
        "BatchNormalization":
            lambda name, layer: save.bn(name, layer),
    }

    model_name = model.predictor.__class__.__name__.lower()
    if not exists(model_name):
        os.makedirs(model_name)

    for layer in model.predictor.links(skipself=True):
        layer_class = layer.__class__.__name__
        if layer_class in save_func:
            save_func[layer_class](model_name, layer)
        else:
            print("{}({}) pass".format(layer.__class__.__name__, layer.name))

