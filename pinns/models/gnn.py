import tensorflow as tf
# import math
from keras import Model
from keras.layers import Input, Dense


class GNN(Model):

    def __init__(self, x_lb=0, x_rb=0, p_lb=0, p_rb=0, hidden_units=None, name="dense_mixed", **kwargs):
        super(GNN, self).__init__(name=name, **kwargs)
        if hidden_units is None:
            self.hidden_units = [100, 100, 100]
        else:
            self.hidden_units = hidden_units

        self.a = float(x_lb)
        self.b = float(x_rb)

        self.psi_a = float(p_lb)
        self.psi_b = float(p_rb)

        # self.pi = tf.constant(math.pi)

        # Initializer
        xavier_init = 'glorot_normal'

        # Model layers
        inp = Input(shape=(1, ), name='input')

        x = Dense(self.hidden_units[0], activation=tf.math.sin, use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'ld0-sin-{self.hidden_units[0]}n')(inp)

        for i in range(1, len(self.hidden_units[1:]) + 1):
            if i % 2 == 0:
                x = Dense(self.hidden_units[i], activation=tf.math.sin, use_bias=True, kernel_initializer=xavier_init,
                          bias_initializer='zeros', name=f'ld{i}-sin-{self.hidden_units[i]}n')(x)
            else:
                x = Dense(self.hidden_units[i], activation=tf.math.cos, use_bias=True, kernel_initializer=xavier_init,
                          bias_initializer='zeros', name=f'ld{i}-cos-{self.hidden_units[i]}n')(x)
        
        out = Dense(1, activation='linear', use_bias=True, kernel_initializer=xavier_init, bias_initializer='zeros',
                    name='output-linear')(x)

        self.model = tf.keras.Model(inputs=inp, outputs=out)

        self.model.summary()

    def call(self, inputs):
        # Absolute value - trial solution 
        out = (self.psi_a * (self.b - inputs)) / (self.b - self.a) + \
              (self.psi_b * (inputs - self.a)) / (self.b - self.a) + \
              (inputs - self.a) * (self.b - inputs) * self.model(inputs)

        # Sin Cos - trial solution
        # out = tf.math.cos(self.pi/2*inputs) * self.psi_a + \
        #       tf.math.sin(self.pi/2*inputs) * self.psi_b + \
        #       tf.math.cos(self.pi/2*inputs) * tf.math.sin(self.pi/2*inputs) * self.model(inputs)

        return out

    def get_config(self):
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
