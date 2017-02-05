#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import exists, join

import numpy as np
import cupy

def _save(name, layer):
    if not exists(join(name, layer.name)):
        os.makedirs(join(name, layer.name))

    for param in layer.params():
        filename = join(name, layer.name, param.name+".dat")
        with open(filename, "wb") as f:
            print(filename, param.data.shape)
            if type(param.data) == cupy.core.core.ndarray:
                np.savetxt(f, cupy.asnumpy(param.data.ravel()), fmt="%8.8f")
            else:
                np.savetxt(f, param.data.ravel(), fmt="%8.8f")

def conv(name, layer): _save(name, layer)

def full(name, layer): _save(name, layer)

def bn(name, layer):
    if not exists(join(name, layer.name)):
        os.makedirs(join(name, layer.name))

    params = [ ("gamma", layer.gamma.data)
             , ("beta", layer.beta.data)
             , ("mean", layer.avg_mean)
             , ("var",  layer.avg_var)
             , ("eps",  np.asarray(layer.eps, dtype=np.float32))
             ]

    for param_name, param_data in params:
        filename = join(name, layer.name, param_name+".dat")
        with open(filename, "wb") as f:
            print(filename, param_data.shape)
            if type(param_data) == cupy.core.core.ndarray:
                np.savetxt(f, cupy.asnumpy(param_data.ravel()), fmt="%8.8f")
            else:
                np.savetxt(f, param_data.ravel(), fmt="%8.8f")

