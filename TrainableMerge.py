from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import functools
from keras import initializations

class TrainableMerge(Layer):
    def __init__(self, output_dim, input_dim,**kwargs):
        init = 'glorot_uniform'
        self.output_dim = output_dim
        self.input_dim= input_dim
        self.init = initializations.get(init)
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(TrainableMerge, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(shape=(input_shape[0], self.output_dim),
                                 initializer=self.init,
                                 trainable=True)
        self.W2 = self.add_weight(shape=(input_shape[1], self.output_dim),
                                  initializer=self.init,
                                  trainable=True)
        super(TrainableMerge, self).build()
    def call(self, x, mask=None):
        e1=x[0]
        e2=x[1]
        ''' Dot Product Final_Result=X1*W1+X2*W2'''
        Tensor_1=K.dot(e1, self.W1)
        Tensor_2=K.dot(e2,self.W2)
        inputs=[Tensor_1,Tensor_2]
        result= K.sum(K.concatenate(inputs,axis=self.concat_axis))
        return result

    def get_output_shape_for(self, input_shape):
        print (input_shape)
        batch_size = input_shape[0][0]
        return (batch_size, self.output_dim)
