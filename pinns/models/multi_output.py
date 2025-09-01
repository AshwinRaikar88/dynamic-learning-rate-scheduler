import tensorflow as tf

from keras import Model
from keras.layers import Input, Dense


class MULTI(Model):

    def __init__(self, x_lb, x_rb, p_lb, p_rb, name="multi_output_network", hidden_units=None,  **kwargs):
        super(MULTI, self).__init__(name=name, **kwargs)

        if hidden_units is None:
            hidden_units = [100, 100, 100]

        self.num_outputs = 3
        output_units = []
        
        self.hidden_units = hidden_units
        self.a = float(x_lb)
        self.b = float(x_rb)
        self.p_lb = p_lb
        self.p_rb = p_rb

        self.psi_a = float(self.p_lb)
        self.psi_b = float(self.p_rb)

        # Initializer
        xavier_init = 'glorot_normal'

        inp = Input(shape=(1,), name='input')

        x = Dense(self.hidden_units[0], activation=tf.math.sin, use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'ld0-sin-{self.hidden_units[0]}n')(inp)

        for i in range(1, len(self.hidden_units[1:]) + 1):
            if i % 2 == 0:
                x = Dense(self.hidden_units[i], activation=tf.math.sin, use_bias=True, kernel_initializer=xavier_init,
                          bias_initializer='zeros', name=f'ld{i}-sin-{self.hidden_units[i]}n')(x)
            else:
                x = Dense(self.hidden_units[i], activation=tf.math.cos, use_bias=True, kernel_initializer=xavier_init,
                          bias_initializer='zeros', name=f'ld{i}-cos-{self.hidden_units[i]}n')(x)

        for i in range(self.num_outputs):
            exec(f"out_{i} = Dense(1, activation='linear', use_bias=True, kernel_initializer=xavier_init, bias_initializer='zeros', \
                      name='output-linear-{i}')(x)")
            exec(f"output_units.append(out_{i})")
        
        self.model = tf.keras.Model(inputs=inp, outputs=output_units)
        self.model.summary()

    def call(self, inputs, **kwargs):
        phi_b = (self.psi_a * (self.b - inputs)) / (self.b - self.a) + (self.psi_b * (inputs - self.a)) / (self.b - self.a)
        out_layer = []
        
        for i in range(self.num_outputs):
            out_layer.append(phi_b + (inputs - self.a) * (self.b - inputs) * self.model(inputs)[i])

        return out_layer

    def get_config(self):
        return {"hidden_units": self.hidden_units,
                "a": self.a,
                "b": self.b,
                "psi_a": self.psi_a,
                "psi_b": self.psi_b,
                "no_outputs": self.num_outputs, 
                }

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)
