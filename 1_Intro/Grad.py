__author__ = '@Piyush3dB'
'''
Demonstrate forward and backward propagation of 
on graph together with gradient blocking.
'''
## Import mxnet python module
import sys
sys.path.append("..")
import find_mxnet
import mxnet as mx


## Define computation graph
E   = 2.718281828459045
w0  = mx.symbol.Variable('w0')
x0  = mx.symbol.Variable('x0')
w1  = mx.symbol.Variable('w1')
x1  = mx.symbol.Variable('x1')
w2  = mx.symbol.Variable('w2')
net = 1/(1+(mx.symbol.pow(E, -1*(w0*x0 + w1*x1 + w2))))


## Comment/Uncomment next line for gradient unblocking/blocking
net = mx.sym.BlockGrad(data=net)


data_shape = (1,)
shape = {}
shape["w0"]=data_shape
shape["x0"]=data_shape
shape["w1"]=data_shape
shape["x1"]=data_shape
shape["w2"]=data_shape
#mx.viz.print_summary(net, shape)
mx.viz.print_summary(net)


## Create executor object and copy arguments
c_exec = net.simple_bind(ctx=mx.cpu(),
                         w0 = shape["w0"],
                         x0 = shape["x0"],
                         w1 = shape["w1"],
                         x1 = shape["x1"],
                         w2 = shape["w2"],
                         grad_req='write')
args={}
args['w0'] = mx.nd.ones(1) *  2.0
args['x0'] = mx.nd.ones(1) * -1.0
args['w1'] = mx.nd.ones(1) * -3.0
args['x1'] = mx.nd.ones(1) * -2.0
args['w2'] = mx.nd.ones(1) * -3.0
c_exec.copy_params_from(arg_params = args)

## Forward computation graph
c_exec.forward(is_train=True)

## Backward through computation graph
c_exec.backward(out_grads=mx.nd.ones(1)*1)

## Extract data for printing
args_dict    = dict(zip(args,[o.asnumpy()[0] for o in c_exec.arg_arrays]))
outputs_dict = dict(zip(net.list_outputs(),[o.asnumpy()[0] for o in c_exec.outputs]))
grads_dict   = dict(zip([n for n in args],[o.asnumpy()[0] for o in c_exec.grad_arrays]))

print "Args   : " + str(["%s=%.2f" %(n,o) for n,o in args_dict.iteritems()])
print "Outputs: " + str(["%s=%.2f" %(n,o) for n,o in outputs_dict.iteritems()])
print "Grads  : " + str(["%s=%.2f" %(n,o) for n,o in grads_dict.iteritems()])


