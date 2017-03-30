__author__ = '@Piyush3dB'
'''
Basic bind operations.
'''

## Import mxnet python module
import sys
sys.path.append("..")
import find_mxnet
import mxnet as mx


a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = a + b

# elemental wise times
d = a * b  
# matrix multiplication
e = mx.sym.dot(a, b)   
# reshape
f = mx.sym.Reshape(d+e, shape=(1,4))  
# broadcast
g = mx.sym.broadcast_to(f, shape=(2,4))  


## Shape inference
arg_name = c.list_arguments()  # get the names of the inputs
out_name = c.list_outputs()    # get the names of the outputs
arg_shape, out_shape, _ = c.infer_shape(a=(2,3), b=(2,3))


ex = c.bind(ctx=mx.cpu(), args={'a' : mx.nd.ones([2,3]), 
                                'b' : mx.nd.ones([2,3])})
ex.forward()
print 'number of outputs = %d\nthe first output = \n%s' % (len(ex.outputs), ex.outputs[0].asnumpy())




## Save and reload symbol
print(c.tojson())
c.save('symbol-c.json')
c2 = mx.symbol.load('symbol-c.json')
c.tojson() == c2.tojson()


## Cast
a = mx.sym.Variable('data')
b = mx.sym.Cast(data=a, dtype='float16')
arg, out, _ = b.infer_type(data='float32')
print({'input':arg, 'output':out})

c = mx.sym.Cast(data=a, dtype='uint8')
arg, out, _ = c.infer_type(data='int32')
print({'input':arg, 'output':out})

## Variable sharing
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = mx.sym.Variable('c')
d = a + b * c

data = mx.nd.ones((2,3))*2
ex = d.bind(ctx=mx.cpu(), args={'a':data, 'b':data, 'c':data})
ex.forward()
ex.outputs[0].asnumpy()


#print(net.list_arguments())

## Show simple bind
#http://mxnet.io/tutorials/python/mixed.html
 # There is also function named simple_bind that simplifies this procedure. 
 # This function first inferences the shapes of all free variables by using 
 # the provided data shape, and then allocate and bind data, which can be 
 # accessed by the attribute arg_arrays of the returned executor.