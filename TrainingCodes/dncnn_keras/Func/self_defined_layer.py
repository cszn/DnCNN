from keras import backend as K
from keras.engine.topology import Layer

class AdLayer(Layer):

    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(AdLayer, self).__init__(**kwargs)

    def build(self, input_shape):
#         print(type(input_shape),input_shape)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1:]),
                                      initializer='uniform',
                                      trainable=True)
        super(AdLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x*self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape