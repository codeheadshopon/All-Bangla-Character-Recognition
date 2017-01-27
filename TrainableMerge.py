from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import functools
from keras import initializations

class TrainableMerge(Layer):
    def __init__(self, output_dim, input_dim,**kwargs):
        init = 'one'
        self.output_dim = output_dim
        self.input_dim= input_dim
        self.init = initializations.get(init)
        print(input_dim)
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(TrainableMerge, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(shape=(1,1),
                                 initializer=self.init,
                                 trainable=True)
        self.W2 = self.add_weight(shape=(1, 1),
                                  initializer=self.init,
                                  trainable=True)
    def call(self, x, mask=None):
        e1=x[0]
        e2=x[1]
        ''' Dot Product Final_Result=X1*W1+X2*W2'''
        Tensor_1=K.dot(e1, self.W1)
        Tensor_2=K.dot(e2,self.W2)
        print(TrainableMerge.get_output_shape_for(self,Tensor_1))
        print(TrainableMerge.get_output_shape_for(self,Tensor_2))

        inputs=[Tensor_1,Tensor_2]
        result = inputs[0]
        for i in range(0, 2):
            result += inputs[i]
        print(TrainableMerge.get_output_shape_for(self,result))
        return result

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
