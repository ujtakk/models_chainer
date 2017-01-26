#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chainer

def dataset(name):
    from chainer.datasets import get_mnist
    from chainer.datasets import get_cifar10
    from chainer.datasets import get_cifar100

    if name == 'mnist':
        print('using MNIST dataset.')
        image_colors = 1
        class_labels = 10
        train, test = get_mnist(ndim=3)
    elif name == 'cifar10':
        print('using cifar10 dataset.')
        image_colors = 3
        class_labels = 10
        train, test = get_cifar10()
    elif name == 'cifar100':
        print('Using CIFAR100 dataset.')
        image_colors = 3
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')

    return {
        "attr": (image_colors, class_labels),
        "data": (train, test)
    }

def model(arch_name, image_colors, class_labels):
    import chainer.links as L
    import models as M

    archs = {
        'vgg': M.VGG,
        'leblock': M.LeBlock,
        'resblock': M.ResBlock,
        'resnet50': M.ResNet50,
        'resnet101': M.ResNet101,
        'resnet152': M.ResNet152,
    }

    return L.Classifier(archs[arch_name](image_colors, class_labels))

def trainer(args, model, optimizer, train, test):
    from chainer import training
    from chainer.training import extensions

    class TestModeEvaluator(extensions.Evaluator):
        def evaluate(self):
            model = self.get_target('main')
            model.train = False
            ret = super(TestModeEvaluator, self).evaluate()
            model.train = True
            return ret

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter  = chainer.iterators.SerialIterator(test, args.batchsize,
                                                  repeat=False, shuffle=False) 
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(TestModeEvaluator(test_iter, model, device=args.gpu))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

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

