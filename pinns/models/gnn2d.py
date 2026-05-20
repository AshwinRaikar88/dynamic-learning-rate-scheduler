import tensorflow as tf

from keras import Model
from keras.layers import Input, Dense
# from keras.initializers import Constant, RandomNormal, HeUniform
from keras.initializers import HeUniform

class GNN2D(Model):

    def __init__(self, name="neural_network_2d", hidden_units=None, **kwargs):
        super(GNN2D, self).__init__(name=name, **kwargs)
        if hidden_units is None:
            hidden_units = [100, 100, 100]
        
        self.hidden_units = hidden_units

        # Initializer
        # xavier_init = 'glorot_normal'
        # xavier_init = Constant(value=0.5)
        # xavier_init = RandomNormal(mean=0.5, stddev=0.5)
        xavier_init = HeUniform()
      
        # Layers
        inp = Input(shape=(2, ), name='input')

        x = Dense(self.hidden_units[0], activation=tf.sin, use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'ld0-sin-{self.hidden_units[0]}n')(inp)
     
        for i in range(1, len(self.hidden_units[1:]) + 1):
            if i % 2 == 0:
                x = Dense(self.hidden_units[i], activation=tf.sin, use_bias=True, kernel_initializer=xavier_init,
                          bias_initializer='zeros', name=f'ld{i}-sin-{self.hidden_units[i]}n')(x)
            else:
                x = Dense(self.hidden_units[i], activation=tf.cos, use_bias=True, kernel_initializer=xavier_init,
                          bias_initializer='zeros', name=f'ld{i}-cos-{self.hidden_units[i]}n')(x)

        out = Dense(1, activation='linear', use_bias=True, kernel_initializer=xavier_init, bias_initializer='zeros',
                    name='output-linear')(x)

        self.model = tf.keras.Model(name=name, inputs=inp, outputs=out)

        # tf.keras.utils.plot_model(self.model, to_file='./model.png', show_shapes=True, show_dtype=False,
        #                           show_layer_names=True,
        #                           rankdir='TB',
        #                           expand_nested=False,
        #                           dpi=96,
        #                           layer_range=None,
        #                           show_layer_activations=False)

        self.model.summary()
        
    def call(self, x, y):
        # Concatenate the two tensors along the second dimension to create a (batch_size, 2) tensor
        return self.model(tf.concat([x, y], axis=1))

    def get_config(self):
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
