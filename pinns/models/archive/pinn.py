import tensorflow as tf
import numpy as np
import cvnn
import cvnn.layers as complex_layers

from keras.losses import MeanSquaredError


class PINN_Acoustic(tf.keras.Model):

    def __init__(self, **kwargs):
        super(PINN_Acoustic, self).__init__(**kwargs)
        self.mse = MeanSquaredError()
        # xavier_init = tf.keras.initializers.GlorotNormal()
        cnn_xavier_init = cvnn.initializers.ComplexGlorotNormal()

        def custom_activation(x):
            return tf.math.exp(x * 1j)

        model = tf.keras.models.Sequential()
        model.add(complex_layers.ComplexInput(input_shape=(1,)))
        model.add(
            complex_layers.ComplexDense(100, activation='cart_tanh', use_bias=True, kernel_initializer=cnn_xavier_init))
        model.add(
            complex_layers.ComplexDense(100, activation='cart_tanh', use_bias=True, kernel_initializer=cnn_xavier_init))
        model.add(
            complex_layers.ComplexDense(100, activation='cart_tanh', use_bias=True, kernel_initializer=cnn_xavier_init))
        model.add(complex_layers.ComplexDense(1, name='output', activation='linear', use_bias=True,
                                              kernel_initializer=cnn_xavier_init))
        # model.add(complex_layers.ComplexDense(1, activation='convert_to_real_with_abs', use_bias=True, kernel_initializer=cnn_xavier_init))

        self.model = model
        self.model.summary()
        self.model.compile()

    # def __call__(self, input):
    #     return self.model(input, self.get_layer['output'].outputs)

    # def custom_loss(self, data):
    #     data = tf.cast(data, dtype=tf.float32)

    #     x = tf.math.abs(data)
    #     y_pred = tf.math.abs(self.model(data))

    #     return self.mse(x, y_pred)

    def custom_loss(self, x):
        x = tf.cast(x, dtype=tf.complex64)
        zeros_tensor = tf.zeros_like(x)
        ones_tensor = tf.ones_like(x)
        two_pi = 2 * np.pi

        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            pred = self.model(x)
            nn_pred = (two_pi - x) / two_pi + (x / two_pi) * pred

            dy_dt = gg.gradient(nn_pred, x)

        ode = (dy_dt * -1j) - nn_pred

        ode_r = tf.math.real(ode)
        ode_i = tf.math.imag(ode)

        x_real = tf.math.real(zeros_tensor)
        x_imag = tf.math.imag(zeros_tensor)

        L1r = self.mse(ode_r, x_real)
        L1i = self.mse(ode_i, x_imag)

        # L1 = self.mse(ode, zeros_tensor)

        # L1r = self.mse(y_real, x_real)
        # L1i = self.mse(y_imag, x_imag)

        # L2 = self.mse(self.model(zeros_tensor)-ones_tensor, zeros_tensor)
        # L2 = tf.cast(L2, dtype=tf.float64)
        loss_total = L1r + L1i
        # loss_total = L1

        # print(L1)
        # print(L2)

        return loss_total

    def train_step(self, x):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.custom_loss(x)
            trainable_variables = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return {"loss": loss}
