import tensorflow as tf
import numpy as np

from keras.layers import Input, Dense
from keras.losses import MeanSquaredError


class PinnAcoustic(tf.keras.Model):

    def __init__(self, frequency=100, num_int_col_points=1000, layers=[100, 100, 100]):
        super().__init__()
        self.model_name = f"TH100x{len(layers)}"
        self.model = None
        self.count = 0

        # Variables associated with physics
        self.f = frequency  # Frequency
        self.c = 340  # Speed of sound
        self.x_lb = 0  # Left boundary
        self.x_rb = 1  # Right boundary
        self.p_lb = 1  # Pressure at the left boundary
        self.p_rb = -1  # Pressure at the right boundary

        # Variables associated with the training
        self.numIntColPoints = num_int_col_points  # Number of internal collocation points
        self.Layers = layers

        self.learning_rate = 0.001  # Learning rate
        self.mse = MeanSquaredError()

        # Wave number
        self.k = 2 * np.pi * self.f / self.c

        vector_ones = tf.ones([1, 1])
        self.X_lb = vector_ones * self.x_lb
        self.X_rb = vector_ones * self.x_rb
        self.P_lb = vector_ones * self.p_lb
        self.P_rb = vector_ones * self.p_rb

        self.source_function = tf.zeros([1, 1])

        # Optimizer
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        self._init_model()

    def _init_model(self):
        plot = True

        xavier_init = tf.keras.initializers.GlorotNormal()

        inp = Input(shape=(1,), name='Input')

        x = Dense(self.Layers[0], activation='tanh', use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros',
                  name=f'Dense-tanh-0')(inp)

        for i in range(1, len(self.Layers[1:]) + 1):
            x = Dense(self.Layers[i], activation='tanh', use_bias=True, kernel_initializer=xavier_init,
                      bias_initializer='zeros', name=f'Dense-tanh-{i}')(x)

        out = Dense(1, activation='linear', use_bias=True, kernel_initializer=xavier_init,
                    bias_initializer='zeros',
                    name='Dense-Output')(x)

        self.model = tf.keras.Model(inputs=inp, outputs=out)
        self.model.summary()
        self.model.compile()

        if plot:
            tf.keras.utils.plot_model(self.model, to_file=f'models/diagrams/{self.model_name}.png', show_shapes=True,
                                      show_dtype=False,
                                      show_layer_names=True,
                                      rankdir='TB',
                                      expand_nested=False,
                                      dpi=96,
                                      layer_range=None,
                                      show_layer_activations=False)

    def custom_loss(self, x):
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            G_x = (1 - x) - x + (1 - x) * x * self.model(x)
            dG_dx = gg.gradient(G_x, x)
        d2G_dx2 = gg.gradient(dG_dx, x)

        diff_eqn = d2G_dx2 + self.k ** 2 * G_x

        return self.mse(self.source_function, diff_eqn)

    def train_step(self, x):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.custom_loss(x)
            trainable_variables = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return {"loss": loss}
