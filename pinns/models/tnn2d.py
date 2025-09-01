import tensorflow as tf

from keras import Model
from keras.layers import Input, Dense


class TNN2D(Model):

    def __init__(self, name="neural_network_2d", hidden_units=None, p_n=0.0, p_s=0.0, p_w=1.0, p_e=-1.0,  **kwargs):
        super(TNN2D, self).__init__(name=name, **kwargs)
        if hidden_units is None:
            hidden_units = [100, 300, 500]
        
        self.hidden_units = hidden_units

        self.eps = tf.constant(1e-54, dtype=tf.float32)
        self.p_n = tf.constant(p_n, dtype=tf.float32)
        self.p_s = tf.constant(p_s, dtype=tf.float32)
        self.p_w = tf.constant(p_w, dtype=tf.float32)
        self.p_e = tf.constant(p_e, dtype=tf.float32)

        # Initializer
        xavier_init = 'glorot_normal'

        # Layers
        inp = Input(shape=(2, ), name='input')

        x = Dense(4, activation='tanh', use_bias=False, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'pre-tanh-4n')(inp)
        
        x = Dense(self.hidden_units[0], activation='tanh', use_bias=True, kernel_initializer=xavier_init,
                  bias_initializer='zeros', name=f'ld0-tanh-{self.hidden_units[0]}n')(x)
        
        for i in range(1, len(self.hidden_units[1:]) + 1):
            x = Dense(self.hidden_units[i], activation='tanh', use_bias=True, kernel_initializer=xavier_init,
                    bias_initializer='zeros', name=f'ld{i}-tanh-{self.hidden_units[i]}n')(x)
            
        # x = Dense(self.hidden_units[-1], activation='tanh', use_bias=True, kernel_initializer=xavier_init,
        #          bias_initializer='zeros', name=f'last-tanh-{self.hidden_units[-1]}n')(x)
        
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

    def call(self, x, y):
        # Concatenate the two tensors along the second dimension to create a (batch_size, 2) tensor
        inputs = tf.concat([x, y], axis=1)

        # Phi_1 = self.line_phi(-1 + self.eps, 1 + self.eps, 1 - self.eps, 1 - self.eps, x, y)
        # Phi_2 = self.line_phi(1 - self.eps, 1 - self.eps, 1 - self.eps, -1 - self.eps, x, y)
        # Phi_3 = self.line_phi(1 - self.eps, -1 - self.eps, -1 + self.eps, -1 + self.eps, x, y)
        # Phi_4 = self.line_phi(-1 + self.eps, -1 + self.eps, -1 + self.eps, 1 + self.eps, x, y)

        Phi_1 = self.line_phi(-1+self.eps, 1, 1-self.eps, 1, x, y)
        Phi_2 = self.line_phi(1, 1-self.eps, 1, -1+self.eps, x, y)
        Phi_3 = self.line_phi(1-self.eps, -1, -1+self.eps, -1, x, y)
        Phi_4 = self.line_phi(-1, -1+self.eps, -1, 1-self.eps, x, y)

        W = ((Phi_1 * Phi_2 * Phi_3) + (Phi_2 * Phi_3 * Phi_4) + (Phi_3 * Phi_4 * Phi_1) + (Phi_4 * Phi_1 * Phi_2))

        W_1 = (Phi_2 * Phi_3 * Phi_4) / W
        W_2 = (Phi_1 * Phi_3 * Phi_4) / W
        W_3 = (Phi_2 * Phi_1 * Phi_4) / W
        W_4 = (Phi_2 * Phi_3 * Phi_1) / W

        return (W_1 * self.p_n) + (W_2 * self.p_e) + (W_3 * self.p_s) + (W_4 * self.p_w) + (W_1 * W_2 * W_3 * W_4 * self.model(inputs))
        

    @staticmethod
    def line_phi(x1, y1, x2, y2, X, Y):
        xc = (x1+x2)/2
        yc = (y1+y2)/2

        L = tf.math.sqrt((x2-x1)**2 + (y2-y1)**2)

        f = ((X-x1) * (y2-y1) - (Y-y1) * (x2-x1)) / L

        t = ((L/2)**2 - ((X - xc)**2 + (Y - yc)**2)) / L

        psi = tf.math.sqrt(t**2 + f**4)

        phi = tf.math.sqrt(f**2 + ((psi - t)/2)**2)

        return phi

        # return tf.round(phi * 1e32) / 1e32

    # def get_config(self):
    #     return {"hidden_units": self.hidden_units}

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)
