__author__ = '@Piyush3dB'
'''
Demonstrate convolution as implemented.
'''
## Import mxnet python module
import sys
sys.path.append("..")
import find_mxnet
import mxnet as mx
import pdb as pdb

## Define computation graph
net  = mx.symbol.Convolution(data=mx.sym.Variable('I'), 
                             num_filter=1, 
                             kernel=(3,3), 
                             name="K")

## Create executor by binding shapes
c_exec = net.simple_bind(ctx=mx.cpu(), I=(1,1,5,5))

## Copy input arguments to executor
args={}
args['I'] = mx.nd.array([1,2,2,1,0,
                         3,3,3,3,0,
                         3,0,0,2,2,
                         0,3,0,3,3,
                         2,2,2,3,2]).reshape((1,1,5,5))
args['K_weight'] = mx.nd.array([0,1,2,
                                3,4,5,
                                6,7,8]).reshape((1,1,3,3))
c_exec.copy_params_from(arg_params=args)

## Print arguments
print c_exec.arg_dict['I'].asnumpy()
print c_exec.arg_dict['K_weight'].asnumpy()

## Forward arguments through computation graph
c_exec.forward()

## Show output
print c_exec.outputs[0].asnumpy()

# Result ------>
#
#[[[[ 60.  56.  52.]
#   [ 39.  61.  66.]
#   [ 54.  78.  82.]]]]#
