__author__ = '@Piyush3dB'
'''
Demonstrate summary of computation graph.
'''

## Import mxnet python module
import sys
sys.path.append("..")
import find_mxnet
import mxnet as mx

## Define computation graph
data  = mx.sym.Variable('data')
bias  = mx.sym.Variable('fc1_bias')
conv1 = mx.symbol.Convolution( data=data , num_filter=32, kernel=(3,3), stride=(2,2))
bn1   = mx.symbol.BatchNorm(   data=conv1)
act1  = mx.symbol.Activation(  data=bn1  , act_type="relu")
mp1   = mx.symbol.Pooling(     data=act1 , kernel=(2,2), stride=(2,2), pool_type='max')
fc1   = mx.sym.FullyConnected( data=mp1  , bias=bias, num_hidden=10)
fc2   = mx.sym.FullyConnected( data=fc1  , num_hidden=10, attr={'lr_mult': '0.00'})
sc1   = mx.symbol.SliceChannel(data=fc2  , num_outputs=10, squeeze_axis=0)


## Print summary
mx.viz.print_summary(sc1)
## Should print:
# ____________________________________________________________________
# Layer (type)              Output Shape    Param #     Previous Layer
# ====================================================================
# data(null)                                0                         
# ____________________________________________________________________
# conv1(Convolution)                        32          data          
# ____________________________________________________________________
# bn1(BatchNorm)                            0           conv1         
# ____________________________________________________________________
# relu1(Activation)                         0           bn1           
# ____________________________________________________________________
# mp1(Pooling)                              0           relu1         
# ____________________________________________________________________
# fc1(FullyConnected)                       0           mp1           
# ____________________________________________________________________
# fc2(FullyConnected)                       0           fc1           
# ____________________________________________________________________
# slice_1(SliceChannel)                     0           fc2           
# ====================================================================
# Total params: 32                                                    
# ____________________________________________________________________


## Print summary with input shape defined
shape = {}
shape["data"]=(1,3,28,28)
mx.viz.print_summary(sc1, shape)
## Should print:
# ____________________________________________________________________
# Layer (type)              Output Shape    Param #     Previous Layer
# ====================================================================
# data(null)                3x28x28         0                         
# ____________________________________________________________________
# conv1(Convolution)        32x13x13        896         data          
# ____________________________________________________________________
# bn1(BatchNorm)            32x13x13        64          conv1         
# ____________________________________________________________________
# relu1(Activation)         32x13x13        0           bn1           
# ____________________________________________________________________
# mp1(Pooling)              32x6x6          0           relu1         
# ____________________________________________________________________
# fc1(FullyConnected)       10              352         mp1           
# ____________________________________________________________________
# fc2(FullyConnected)       10              110         fc1           
# ____________________________________________________________________
# slice_1(SliceChannel)                     0           fc2           
# ====================================================================
# Total params: 1422                                                  
# ____________________________________________________________________
# 
