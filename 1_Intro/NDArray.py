__author__ = '@Piyush3dB'
'''
Basic Operations using milti-dimensional (ND) arrays.
'''

## Import mxnet python module
import sys
sys.path.append("..")
import find_mxnet
import mxnet as mx



## Alloc NDArray on CPU or GPU
a = mx.nd.empty((2, 3))           # create a 2-by-3 matrix on cpu
b = mx.nd.empty((2, 3), mx.gpu()) # create a 2-by-3 matrix on gpu 0
print b.shape   # get shape
print b.context # get device info


## NDArray Initialisation
a = mx.nd.zeros((2, 3)) # create a 2-by-3 matrix filled with 0
b = mx.nd.ones( (2, 3)) # create a 2-by-3 matrix filled with 1
b[:] = 2 # set all elements of b to 2


## Convert NDArray to numpy.ndarray
print type(a)
print type(b.asnumpy())
print b.asnumpy()

import numpy as np

## Copy numpy.ndarray to NDArray
a = mx.nd.empty((2, 3))
a[:] = np.random.uniform(-0.1, 0.1, a.shape)
print type(a)
print a.asnumpy()


## NDArray element-wise operations
a = mx.nd.ones((2, 3)) * 2
b = mx.nd.ones((2, 3)) * 4
print a.asnumpy()
print b.asnumpy()
c = a + b
print c.asnumpy()
d = a * b
print d.asnumpy()


## Move and compute on GPU
a = mx.nd.ones((2, 3)) * 2
b = mx.nd.ones((2, 3), mx.gpu()) * 3
c = a.copyto(mx.gpu()) * b
print c.asnumpy()