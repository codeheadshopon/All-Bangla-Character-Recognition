from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import functools
from keras import initializations
import theano.tensor as T
from theano import function
import theano
from keras import regularizers,constraints
class CustomMerge(Layer):
    def __init__(self, output_dim, input_dim=None,**kwargs):
        init = 'uniform'
        W_regularizer = None
        W_constraint = None,
        self.output_dim = output_dim
        self.input_dim= input_dim
        # self.init = initializations.get(init)
        dim_ordering = K.image_dim_ordering()
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(CustomMerge, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W=theano.shared(np.array(0.5,'float32'),'x')

        # super(CustomMerge, self).build()
    def call(self, x, mask=None):
        e1=x
        # Tensor_1=K.dot(e1, self.W)
        Tensor_1=e1*self.W
        return Tensor_1

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
