import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense


class NEU_NN(Model):

    def __init__(self, name="dense_mixed", hidden_units=[100, 100, 100], **kwargs):
        super(NEU_NN, self).__init__(name=name, **kwargs)
        self.hidden_units = hidden_units

        # Initializer
        xavier_init = 'glorot_normal'
        # special_initializer = tf.keras.initializers.RandomNormal(mean=2.0, stddev=0.8)


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
            x = Dense(self.hidden_units[i], activation='linear', use_bias=True, kernel_initializer=xavier_init,
                        bias_initializer='zeros', name=f'ld{i}-linear-{self.hidden_units[i]}n')(x)

        out = Dense(1, activation='linear', use_bias=True, kernel_initializer=xavier_init, bias_initializer='zeros',
                    name='output-linear')(x)

        self.model = tf.keras.Model(inputs=inp, outputs=out, name='NEU_NN')

        self.model.summary()

        self.model.built = True

    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
