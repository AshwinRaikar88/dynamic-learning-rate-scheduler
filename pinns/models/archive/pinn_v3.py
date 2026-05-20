import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from keras.layers import Input, Dense
from keras.losses import MeanSquaredError

mse = MeanSquaredError()
source_function = tf.zeros([1, 1])


@tf.function
def loss_function(model, x, k):
    with tf.GradientTape(persistent=True) as g:  # Forward pass
        g.watch(x)
        G_x = (1 - x) - x + (1 - x) * x * model(x)
        dG_dx = g.gradient(G_x, x)
        d2G_dx2 = g.gradient(dG_dx, x)

        diff_eqn = d2G_dx2 + k ** 2 * G_x
        # loss = tf.reduce_mean(tf.square(source_function - f_pred))
        loss = mse(source_function, diff_eqn)
        return (loss)


def function_factory(model, x, k):
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.

        """

        with tf.GradientTape() as tape:  # Forward pass
            tape.watch(model.trainable_variables)
            assign_new_model_parameters(params_1d)
            loss = loss_function(model, x, k)
            grads = tape.gradient(loss, model.trainable_variables)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tf.dynamic_stitch(idx, grads)

            # store loss value
            tf.py_function(f.loss.append, inp=[loss], Tout=[])

            return loss, grads

    # store these information
    f.assign_new_model_parameters = assign_new_model_parameters
    f.loss = []

    return f


class PinnAcoustic(tf.keras.Model):

    def __init__(self, frequency=100, num_int_col_points=1000, layers=[100, 100, 100]):
        super().__init__()

        # Training parameters
        self.model = None
        self.model_name = f"DNN100x{len(layers)}"
        self.batch_size = 0
        self.learning_rate = 0
        self.epochs = 0
        self.step_count = 0
        self.epoch_count = 0
        self.num_steps = 0

        # Optimizer parameters
        self.mse = MeanSquaredError()
        self.optimizer = None
        self.initial_values = None
        self.func = None

        # Variables associated with physics model
        self.f = frequency  # Frequency
        self.c = 340  # Speed of sound
        self.x_lb = 0  # Left boundary
        self.x_rb = 1  # Right boundary
        self.p_lb = 1  # Pressure at the left boundary
        self.p_rb = -1  # Pressure at the right boundary
        self.k = 2 * np.pi * self.f / self.c  # Wavenumber

        # Variables associated with the training
        self.numIntColPoints = num_int_col_points  # Number of internal collocation points
        self.Layers = layers

        # Tensors
        vector_ones = tf.ones([1, 1])
        self.X_lb = vector_ones * self.x_lb
        self.X_rb = vector_ones * self.x_rb
        self.P_lb = vector_ones * self.p_lb
        self.P_rb = vector_ones * self.p_rb

        self.source_function = tf.zeros([1, 1])

        self._init_model()

    def start_training(self, data, lr=0.0001, batch_size=1000, epochs=10, callbacks=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = lr
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        self.num_steps = self.numIntColPoints // self.batch_size
        self.epoch_count = 0

        self._init_optimizer()

        self.compile(optimizer=self.optimizer, run_eagerly=True)
        # NN.compile(run_eagerly=True)

        self.fit(data, epochs=self.epochs, batch_size=self.batch_size, callbacks=[callbacks])

    def _init_optimizer(self):
        shapes = tf.shape_n(self.model.trainable_variables)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = []  # stitch indices
        part = []  # partition indices

        for i, shape in enumerate(shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n

        part = tf.constant(part)

        self.initial_values = tf.dynamic_stitch(idx, self.model.trainable_variables)

    def _init_model(self):
        plot = True
        # xavier_init = tf.keras.initializers.GlorotNormal()
        xavier_init = tf.keras.initializers.HeNormal()
        inp = Input(shape=1, name='Input')

        x = Dense(self.Layers[0], activation=tf.math.sin, use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros',
                  name=f'Dense-0')(inp)

        for i in range(1, len(self.Layers[1:]) + 1):
            x = Dense(self.Layers[i], activation=tf.math.sin, use_bias=True, kernel_initializer=xavier_init,
                      bias_initializer='zeros',
                      name=f'Dense-sine-{i}')(x)

        # x = Dense(100, activation='tanh', use_bias=True, kernel_initializer=xavier_init, bias_initializer='zeros',
        #           name=f'Dense-tanh')(x)

        x = Dense(100, activation='tanh', use_bias=True, kernel_initializer=xavier_init, bias_initializer=xavier_init,
                  name=f'Dense-tan-out')(x)

        x = Dense(100, activation="linear", use_bias=True, kernel_initializer=xavier_init, bias_initializer=xavier_init,
                  name=f'Dense-linear')(x)

        out = Dense(1, activation='linear', use_bias=True, kernel_initializer=xavier_init,
                    bias_initializer='zeros', name='Dense-Output')(x)

        self.model = tf.keras.Model(inputs=inp, outputs=out)
        self.model.summary()
        self.model.compile()

        if plot:
            tf.keras.utils.plot_model(self.model, to_file=f'models/diagrams/{self.model_name}.png', show_shapes=True, show_dtype=False,
                                      show_layer_names=True,
                                      rankdir='TB',
                                      expand_nested=False,
                                      dpi=96,
                                      layer_range=None,
                                      show_layer_activations=False)

    def custom_loss(self, x):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            G_x = (1 - x) - x + (1 - x) * x * self.model(x)
            dG_dx = tape.gradient(G_x, x)
        d2G_dx2 = tape.gradient(dG_dx, x)

        diff_eqn = d2G_dx2 + self.k ** 2 * G_x

        return self.mse(self.source_function, diff_eqn)

    def train_step(self, x):
        self.step_count += 1
        if self.step_count == self.num_steps:
            self.epoch_count += 1
            print(f"Epoch {self.epoch_count} completed")
            self.step_count = 0
        # else:
        #     print(f"Step {self.step_count} completed")

        if self.epoch_count > 80:
            if self.func == None:
                self.func = function_factory(self.model, x, self.k)

            results = tfp.optimizer.lbfgs_minimize(
                self.func,
                num_correction_pairs=10,
                initial_position=self.initial_values,
                tolerance=1e-2
            )
            # if results.converged.numpy():
            self.func.assign_new_model_parameters(results.position)
            return {"loss": self.func.loss[-1].numpy()}
        else:
            with tf.GradientTape(persistent=True) as tape:
                loss = self.custom_loss(x)
                trainable_variables = self.model.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))
            return {"loss": loss}
