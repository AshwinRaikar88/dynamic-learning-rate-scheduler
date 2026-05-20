import tensorflow as tf

from keras import Model
from keras.layers import Input, Dense, Dot


class MNN2D(Model):

    def __init__(self, name="multibranch_2d", hidden_units=None, **kwargs):
        super(MNN2D, self).__init__(name=name, **kwargs)
        
        self.model_name = name

        if hidden_units is None:
            self.hidden_units = [100, 100, 100]
        else:
            self.hidden_units = hidden_units

        # Initializer
        xavier_init = 'glorot_normal'

        # Layers
        inp1 = Input(shape=(1, ), name='input1')
        inp2 = Input(shape=(1, ), name='input2')

        x = Dense(50, activation=tf.math.sin, use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'x1-tanh-10n')(inp1)        
        
        x = Dense(100, activation=tf.math.cos, use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'x2-tanh-10n')(x)

        x = Dense(100, activation=tf.math.sin, use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'x3-tanh-10n')(x)
        
        x = Dense(10, activation='tanh', use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'x4-tanh-1n')(x)
        
        y = Dense(50, activation=tf.math.sin, use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'y1-tanh-10n')(inp2)
        
        y = Dense(100, activation=tf.math.cos, use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'y2-tanh-10n')(y)
        
        y = Dense(100, activation=tf.math.sin, use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'y3-tanh-10n')(y)
        
        y = Dense(10, activation='tanh', use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'y4-tanh-1n')(y)
        
        mul_layer = Dot(axes=1)([x, y])
       
        out = Dense(1, activation='linear', use_bias=True, kernel_initializer=xavier_init, bias_initializer='zeros',
                    name='output-linear')(mul_layer)

        self.model = tf.keras.Model(name=self.model_name, inputs=[inp1, inp2], outputs=out)

        tf.keras.utils.plot_model(self.model, to_file=f'./models/diagrams/{self.model_name}.png', show_shapes=True, show_dtype=False,
                                  show_layer_names=True,
                                  rankdir='TB',
                                  expand_nested=False,
                                  dpi=96,
                                  layer_range=None,
                                  show_layer_activations=False)

        self.model.summary()
        
    def call(self, x, y):
        return self.model([x, y])

    def get_config(self):
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
