from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import functools
from keras import initializations
import theano.tensor as T
from theano import function
import theano
class CustomMerge(Layer):
    def __init__(self, output_dim, input_dim,**kwargs):
        init = 'zero'
        self.output_dim = output_dim
        self.input_dim= input_dim
        self.init = initializations.get(init)
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(CustomMerge, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(shape=(input_shape[0], self.output_dim),
                                 initializer=self.init,
                                 trainable=True)
        # self.x=theano.shared(np.array([1],'float32'),'x')

        super(CustomMerge, self).build()
    def call(self, x, mask=None):
        e1=x[0]
        ''' Dot Product Final_Result=X1*W1+X2*W2'''
        Tensor_1=K.dot(e1, self.W1)
        return Tensor_1

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
