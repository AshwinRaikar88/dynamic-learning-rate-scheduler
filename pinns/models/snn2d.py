import tensorflow as tf

from keras import Model
from keras.layers import Input, Dense, Concatenate, Activation, Layer


class SirenLayer(Layer):
    def __init__(self, units, omega_initializer=30.0, use_bias=True, **kwargs):
        super(SirenLayer, self).__init__(**kwargs)
        self.units = units
        self.omega_initializer = omega_initializer
        self.use_bias = use_bias

    def build(self, input_shape):
        
        self.omega = tf.constant(self.omega_initializer)
        
        # self.omega = self.add_weight(
        #     name='omega',
        #     shape=[1,],
        #     initializer=tf.keras.initializers.Constant(self.omega_initializer),
        #     trainable=False
        # )    

        self.siren_weights = self.add_weight(
                name='siren_weights',
                shape=(input_shape[-1], self.units),
                initializer='glorot_normal',  # You can customize the initializer
                trainable=True
            )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='glorot_normal',  # You can customize the initializer
                trainable=True
            )

        super(SirenLayer, self).build(input_shape)

    def call(self, inputs):
        if self.use_bias:
            siren_output = tf.sin(tf.matmul(inputs, self.siren_weights) * self.omega + self.bias)
        else:
            siren_output = tf.sin(tf.matmul(inputs, self.siren_weights) * self.omega)
        return siren_output


class SNN2D(Model):
    def __init__(self, name="neural_network_2d", hidden_units=None, p_n=0.0, p_s=0.0, p_w=1.0, p_e=-1.0,  **kwargs):
        super(SNN2D, self).__init__(name=name, **kwargs)
        if hidden_units is None:
            hidden_units = [100, 100, 100]
        
        self.hidden_units = hidden_units

        self.eps = tf.constant(1e-5)
        self.p_n = tf.constant(p_n)
        self.p_s = tf.constant(p_s)
        self.p_w = tf.constant(p_w)
        self.p_e = tf.constant(p_e)

        # Initializer
        xavier_init = 'glorot_normal'

        # Create a custom activation layer using the siren_activation function
        # siren = Activation(self.siren_activation)

        # Layers
        inp = Input(shape=(2,), name='input')

        x = Dense(self.hidden_units[0], activation=tf.sin, use_bias=True, kernel_initializer=xavier_init,
                    bias_initializer='zeros', name=f'ld0-sin-{self.hidden_units[0]}n')(inp)
        
        x = SirenLayer(self.hidden_units[0], omega_initializer=0.5, name=f'ld{0}-siren-{self.hidden_units[0]}n')(x)

        
        for i in range(1, len(self.hidden_units)):
            x = SirenLayer(self.hidden_units[i], omega_initializer=0.9, name=f'ld{i}-siren-{self.hidden_units[i]}n')(x)
        

        out = Dense(1, activation='linear', use_bias=True, kernel_initializer=xavier_init, bias_initializer='zeros',
                    name='output-linear')(x)
        

        self.model = tf.keras.Model(name="PINN_2D_model",inputs=inp, outputs=out)

        # tf.keras.utils.plot_model(self.model, to_file='./model.png', show_shapes=True, show_dtype=False,
        #                           show_layer_names=True,
        #                           rankdir='TB',
        #                           expand_nested=False,
        #                           dpi=96,
        #                           layer_range=None,
        #                           show_layer_activations=False)

        self.model.summary()

    # @staticmethod
    # def siren_activation(x, omega_0=30.0):
    #     return tf.math.sin(omega_0 * x) 
        

    def call(self, x, y):
        # Concatenate the two tensors along the second dimension to create a (batch_size, 2) tensor
        inputs = tf.concat([x, y], axis=1)

        Phi_1 = self.line_phi(-1 + self.eps, 1 + self.eps, 1 - self.eps, 1 - self.eps, x, y)
        Phi_2 = self.line_phi(1 - self.eps, 1 - self.eps, 1 - self.eps, -1 - self.eps, x, y)
        Phi_3 = self.line_phi(1 - self.eps, -1 - self.eps, -1 + self.eps, -1 + self.eps, x, y)
        Phi_4 = self.line_phi(-1 + self.eps, -1 + self.eps, -1 + self.eps, 1 + self.eps, x, y)

        W = ((Phi_1 * Phi_2 * Phi_3) + (Phi_2 * Phi_3 * Phi_4) + (Phi_3 * Phi_4 * Phi_1) + (Phi_4 * Phi_1 * Phi_2))

        W_1 = (Phi_2 * Phi_3 * Phi_4) / W
        W_2 = (Phi_1 * Phi_3 * Phi_4) / W
        W_3 = (Phi_2 * Phi_1 * Phi_4) / W
        W_4 = (Phi_2 * Phi_3 * Phi_1) / W

        out = (W_1 * self.p_n) + (W_2 * self.p_e) + (W_3 * self.p_s) + (W_4 * self.p_w) + (W_1 * W_2 * W_3 * W_4 * self.model(inputs))
        
        return out

    def line_phi(self, x1, y1, x2, y2, X, Y):
        xc = (x1+x2)/2
        yc = (y1+y2)/2

        L = tf.math.sqrt((x2-x1)**2 + (y2-y1)**2)

        f = ((X-x1) * (y2-y1) - (Y-y1) * (x2-x1)) / L

        t = ((L/2)**2 - ((X - xc)**2 + (Y - yc)**2)) / L

        psi = tf.math.sqrt(t**2 + f**4)

        phi = tf.math.sqrt(f**2 + ((psi - t)/2)**2)
        phi = tf.round(phi * 1e5) / 1e5
        return phi

    # def get_config(self):
    #     return {"hidden_units": self.hidden_units}

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)
