'''
If you did

git clone https://github.com/dmlc/mxnet.git ../mxnet

then set

MY_MXNET_PATH = '../mxnet/python'

Note:  make sure you've built MXNet according to the instructions.
'''

# Specify MXNet python module location
MY_MXNET_PATH = '../mxnet/python'

# Now import the module for use by the example scripts
try:
    import mxnet as mx
except ImportError:
    import os, sys
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, MY_MXNET_PATH))
    import mxnet as mx