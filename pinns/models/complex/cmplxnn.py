import tensorflow as tf

from keras import Model
from keras.layers import Input, Dense

tf.random.set_seed(1234)

class CMPLXNN(Model):

    def __init__(self, x_lb, x_rb, p_lb, p_rb, name="complex_mlp", hidden_units=None, **kwargs):
        super(CMPLXNN, self).__init__(name=name, **kwargs)

        if hidden_units is None:
            hidden_units = [100, 100, 100]

        self.hidden_units = hidden_units
        self.a = float(x_lb)
        self.b = float(x_rb)
        self.p_lb = p_lb
        self.p_rb = p_rb

        self.psi_ar = float(tf.math.real(self.p_lb))
        self.psi_br = float(tf.math.real(self.p_rb))
        self.psi_ai = float(tf.math.imag(self.p_lb))
        self.psi_bi = float(tf.math.imag(self.p_rb))

        # Initializer
        xavier_init = tf.keras.initializers.GlorotNormal(seed=123)
        # xavier_init = 'glorot_normal'

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

        out_r = Dense(1, activation='linear', use_bias=True, kernel_initializer=xavier_init, bias_initializer='zeros',
                      name='output-linear-real')(x)
        out_i = Dense(1, activation='linear', use_bias=True, kernel_initializer=xavier_init, bias_initializer='zeros',
                      name='output-linear-imag')(x)

        self.model = tf.keras.Model(inputs=inp, outputs=[out_r, out_i])
        self.model.summary()

    def call(self, inputs, **kwargs):
        out_r = (self.psi_ar * (self.b - inputs)) / (self.b - self.a) + \
                (self.psi_br * (inputs - self.a)) / (self.b - self.a) + \
                (inputs - self.a) * (self.b - inputs) * self.model(inputs)[0]

        out_i = (self.psi_ai * (self.b - inputs)) / (self.b - self.a) + \
                (self.psi_bi * (inputs - self.a)) / (self.b - self.a) + \
                (inputs - self.a) * (self.b - inputs) * self.model(inputs)[1]

        return out_r, out_i

    def get_config(self):
        return {"hidden_units": self.hidden_units,
                "a": self.a,
                "b": self.b,
                "psi_ar": self.psi_ar,
                "psi_br": self.psi_br,
                "psi_ai": self.psi_ai,
                "psi_bi": self.psi_bi,
                }

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)
