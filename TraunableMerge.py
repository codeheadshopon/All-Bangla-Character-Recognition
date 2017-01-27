#!/usr/bin/python

import scipy.stats as stats

from keras import backend as K
from keras.engine.topology import Layer

class NeuralTensorLayer(Layer):
  def __init__(self, output_dim, input_dim=None, **kwargs):
    self.output_dim = output_dim #k
    self.input_dim = input_dim   #d
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(NeuralTensorLayer, self).__init__(**kwargs)


  def build(self, input_shape):

    print("Check 0")
    k = self.output_dim
    d = self.input_dim
    initial_W_values = 0.5
    initial_V_values = 0.5
    self.W = K.variable(initial_W_values)
    self.V = K.variable(initial_V_values)
    self.b = K.zeros((self.input_dim,))
    self.trainable_weights = [self.W, self.V, self.b]

  def call(self, inputs, mask=None):
    if type(inputs) is not list or len(inputs) <= 1:
      raise Exception('BilinearTensorLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
    print("Check 7")
    e1 = inputs[0]
    e2 = inputs[1]
    batch_size = K.shape(e1)[0]

    k = self.output_dim
    # print([e1,e2])
    print("CHeck 8")
    feed_forward_product = K.dot(K.concatenate([e1,e2]), self.V)
    print(feed_forward_product)
    print("CHeck 9")
    bilinear_tensor_products = [ K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1) ]
    print(bilinear_tensor_products)
    print("Check 5")
    for i in range(k)[1:]:
      btp = K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
      bilinear_tensor_products.append(btp)
    print("Check 6")
    result = K.tanh(K.reshape(K.concatenate(bilinear_tensor_products, axis=0), (batch_size, k)) + feed_forward_product)
    print(result)
    return result


  def get_output_shape_for(self, input_shape):
    print (input_shape)
    batch_size = input_shape[0][0]
    print('aha',batch_size)
    return (batch_size, self.output_dim)