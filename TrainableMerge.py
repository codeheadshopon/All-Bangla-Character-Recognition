from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import functools
from keras import initializations
import theano.tensor as T
from theano import function
class CustomMerge(Layer):
    def __init__(self, output_dim, input_dim,**kwargs):
        init = 'glorot_uniform'
        self.output_dim = output_dim
        self.input_dim= input_dim
        self.init = initializations.get(init)
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(CustomMerge, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.W1 = self.add_weight(shape=(input_shape[0], self.output_dim),
        #                          initializer=self.init,
        #                          trainable=True)
        # self.W2 = self.add_weight(shape=(input_shape[1], self.output_dim),
        #                           initializer=self.init,
        #                           trainable=True)
        x = T.dscalar('x')
        z = x
        f = function([x], z)

        self.W1=f(0.5)
        self.W2=f(0.5)
        super(CustomMerge, self).build()
    def call(self, x, mask=None):
        e1=x[0]
        e2=x[1]
        ''' Dot Product Final_Result=X1*W1+X2*W2'''
        Tensor_1=K.dot(e1, self.W1)
        Tensor_2=K.dot(e2,self.W2)
        result= K.sum(K.concatenate([Tensor_1, Tensor_2]))
        return result

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
