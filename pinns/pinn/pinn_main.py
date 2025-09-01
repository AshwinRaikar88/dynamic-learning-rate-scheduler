"""
Author: Ashwin A Raikar
Description: Physics Informed Neural Network PINN
            Solving Helmholtz Equation in 1D using trial solution method

Date Modified: July 2024
"""

import os
import numpy as np
import tensorflow as tf

from models.gnn import GNN
from keras.losses import MeanSquaredError

from utils.dataset_utils import *
from utils.callbacks import DLRS

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

np.random.seed(1234)
tf.random.set_seed(1234)

# supress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# disable GPU devices
# tf.config.set_visible_devices([], 'GPU')

# enable GPU devices
# tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class PINN:
    def __init__(self, configs=None):
        # Variables associated with physics
        self.f = configs[0]['f']
        self.c = configs[0]['c']
        self.x_lb = configs[0]['x_lb']
        self.x_rb = configs[0]['x_rb']
        self.p_lb = configs[0]['p_lb']
        self.p_rb = configs[0]['p_rb']
        self.k = 2 * np.pi * self.f / self.c

        self.numIntColPoints = configs[1]['numIntColPoints']

        # initialize the model
        if configs[2]['pretrained_path'] is None:
            print("Initializing GNN model...")
            self.model = GNN(self.x_lb, self.x_rb, self.p_lb, self.p_rb,
                             configs[2]['layers'], configs[2]['name'])
        else:
            print("Loading pretrained GNN model...")
            if os.path.exists(configs[2]['pretrained_path']):
                self.model = self.load_model(configs[2]['pretrained_path'])
            else:
                print("model weights not found")

        self.optimizer = None
        self.mse = MeanSquaredError()

        # Variables associated with the training
        self.exp_name = configs[1]['exp_name']
        self.train_data = None
        self.len_train_dataset = 0
        self.batch_size = configs[1]['batch_size']

        # Variables associated with Callbacks
        self.train_summary_writer = None
        self.update_num = configs[1]['epoch_verbose']
        self.stop_training = False

        self.lr = configs[1]['lr']
        self.lr_upper_threshold = 9.0e-3
        self.lr_lower_threshold = 5.55e-5
        self.loss_metrics = []
        self.prev_loss = 0.0
        self.best_loss = 0.9

        self.output_dir = configs[1]['output_images_path']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        data = generate_data(x_lb=self.x_lb, x_rb=self.x_rb, nc=self.numIntColPoints)
        self.load_data(data, self.batch_size)

    def load_data(self, data, batch_size):
        # Create mini batches
        self.len_train_dataset = len(data)
        self.train_data = self.create_batches(data, batch_size)
        self.train_data = tf.convert_to_tensor(self.train_data, dtype=tf.float32)
        print("Dataset loaded successfully")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)

    def save(self, path="./weights/"):
        self.model.save(path)
        print(f"Model saved in: {path}")

    @staticmethod
    def create_batches(data, batch_size):
        output = []
        num_items = len(data)
        num_batches = num_items // batch_size

        lb = 0
        rb = batch_size

        for i in range(num_batches):
            output.append(data[lb:rb])
            lb = rb
            rb += batch_size
            if lb == num_items:
                return output

        # Append remaining items
        output.append(data[lb:])

        # Append zeros for remaining items
        for i in range(batch_size - len(data[lb:])):
            output[-1].append([0.0])

        return output

    def custom_loss(self, x):
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            P_x = self.model(x)
            dP_dx = gg.gradient(P_x, x)
        d2P_dx2 = gg.gradient(dP_dx, x)
        diff_eqn = d2P_dx2 + self.k ** 2 * P_x
        del gg
        return self.mse(0.0, diff_eqn)

    def train(self, epochs, learning_rate=0.005, optimizer='adam', dynamic_lr=[None]):
        """
        Custom training loop

        :param epochs: Number of epochs to train (int)
        :param optimizer: Optimization algorithm (obj)
        :param learning_rate: Learning rate (float)
        :param dlrs: Tuple containing delta_i, delta_d values to enable or disable callback for dynamic learning rate
        :return:
        """

        self.lr = learning_rate

        scheduler = None
        if dynamic_lr[0] == 'dlrs-v2':
            scheduler = DLRS(self.lr_upper_threshold, self.lr_lower_threshold,
                             delta_increase=dynamic_lr[1], delta_decrease=dynamic_lr[2])
        
        elif dynamic_lr[0] == 'dlrs-v3':
            scheduler = DLRS(self.lr_upper_threshold, self.lr_lower_threshold)
        
        elif dynamic_lr[0] == 'exponential':
            initial_learning_rate = learning_rate
            final_learning_rate = 0.0005
            learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
            steps_per_epoch = 10

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=initial_learning_rate,
                            decay_steps=steps_per_epoch,
                            decay_rate=learning_rate_decay_factor,
                            staircase=True)

        if optimizer == 'sgd':
            print("OPTIMIZER: Stochastic Gradient Descent")
            self.optimizer = tf.optimizers.SGD(self.lr)
        
        elif optimizer == 'adam':
            print("OPTIMIZER: Adam")
            
            if dynamic_lr == 'exponential':
                self.optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
            else:
                self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        else:
            print("OPTIMIZER: Invalid choose between adam, sgd")
            return 0

        # initialize Tensorboard    
        train_log_dir = f'logs/{self.exp_name}/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        total_batches = len(self.train_data)

        print(f"Starting training for {epochs} epochs\n-----------------------------")
        for epoch in range(epochs + 1):
            loss = 0
            self.loss_metrics = []

            if self.stop_training:
                print("Stopping training due to early stopping callback.")
                break

            if epoch % self.update_num == 0:
                print(f"Batch\t| Loss\t| Progress")
                print("----" * 8)

            for batch_num, batch in enumerate(self.train_data):
                with tf.GradientTape() as tape:
                    tape.watch(batch)
                    loss = self.custom_loss(batch)
                    trainable_variables = self.model.trainable_variables

                gradients = tape.gradient(loss, trainable_variables)
                
                # clip gradients by norm
                # gradients = [(tf.clip_by_norm(grad, clip_norm=1.0)) for grad in gradients]
        
                self.optimizer.apply_gradients(zip(gradients, trainable_variables))
                self.loss_metrics.append(loss.numpy())

                if epoch % self.update_num == 0:
                    print(f"  {batch_num+1}\t| {loss.numpy():.2f}\t| {((batch_num+1)/total_batches) * 100:.1f}%")
            
            if epoch % 50 == 0:
                self.stop_training = self.plot_output(epoch)
                
            # DLRS 
            if dynamic_lr[0] == 'dlrs':
                self.callback_dynamic_lr(epoch)
            
            elif dynamic_lr[0] == 'dlrs-v2':
                self.lr, _ = scheduler.step(self.lr, self.loss_metrics)
                self.optimizer.learning_rate.assign(self.lr)

            elif dynamic_lr[0] == 'dlrs-v3':
                self.lr, _ = scheduler.step_v3(self.lr, self.loss_metrics)
                self.optimizer.learning_rate.assign(self.lr)
                
            # Update loss metrics
            self.callback_tensorboard(epoch)

            # Custom Early stopping
            # self.callback_early_stopping(epoch, loss)
        
            # print(f"Learning rate: {self.lr}")

        # Save model after finishing training
        self.model.save(f"./weights/{self.exp_name}-last.h5")

    def callback_early_stopping(self, epoch, loss, save_best_weights=False):
        if np.isnan(loss) or (loss.numpy() > 10000 * self.prev_loss and epoch > 100):
            print(f"Loss exploded from {self.prev_loss} to {loss.numpy()}")
            self.stop_training = True

        else:
            self.prev_loss = loss.numpy()

        # Save best weights based on loss values
        if save_best_weights and epoch > 100 and (loss.numpy() < self.best_loss):
            self.best_loss = loss.numpy()
            try:
                print(f"Saving best weights")
                save_path = f"./weights/{self.exp_name}-{self.best_loss:.3f}.h5"
                self.model.save(save_path)
                print(f"Weights saved in:{save_path}")
                self.stop_training = True

            except Exception as ex:
                print(f"Unable to save weights:\nException {ex}")

        
    def callback_tensorboard(self,epoch):
        m = tf.keras.metrics.Mean()
        m.update_state(self.loss_metrics)

        if epoch % self.update_num == 0:
            print("----" * 8)
            print(f"Summary: Epoch {epoch}")
            print("----" * 8)
            print(f"Learning rate: {self.lr:.6f}\nTotal loss: {m.result().numpy():.3f}")
            print("----" * 8 + "\n")

        self.lr = self.optimizer.learning_rate.numpy()

        with self.train_summary_writer.as_default():
            tf.summary.scalar(name="Train/loss", data=m.result().numpy(), step=epoch)
            tf.summary.scalar(name="Params/Learning rate", data=self.lr, step=epoch)
        self.train_summary_writer.flush()
        m.reset_state()
        
    def callback_dynamic_lr(self, epoch):
        if self.lr > self.lr_lower_threshold:
            # Calculate mean of loss values
            Lm = np.mean(self.loss_metrics)

            # Calculate normalized slope
            Norm = (self.loss_metrics[-1] - self.loss_metrics[0]) / Lm
            # print(f"{epoch}. Norm = {Norm}, Learning rate = {self.lr}")

            if Norm == 0:
                Norm = 1

            try:
                # this function gets the exponential index of a number
                n = np.floor(np.log10(abs(Norm)))
            except Exception as Ex:
                print(Ex)
                Norm = 1
                n = np.floor(np.log10(abs(Norm)))

            # Higher the magnitue less the impact
            if Norm >= 1:
                # print(f"Norm = {Norm} Up, Learning rate= {self.lr}")
                Norm *= np.pow(10, n - 4)  # n-2
                self.lr -= Norm

            elif Norm < 0:
                # print(f"Norm = {Norm} Dn, Learning rate= {self.lr}")
                Norm *= np.pow(10, n - 5)
                self.lr -= Norm

            else:
                # print(f"Norm = {Norm} Up, Learning rate= {self.lr}")
                Norm *= np.pow(10, n - 3)  # n-2
                self.lr -= Norm

            self.lr = abs(self.lr)

            self.optimizer.learning_rate.assign(self.lr)
            # print(f"{epoch}. New Norm = {Norm}, New Learning rate = {self.lr}")

        elif self.lr < self.lr_lower_threshold:
            self.lr = self.lr_lower_threshold
            self.optimizer.learning_rate.assign(self.lr)

        else:
            return 0

    def plot_output(self, epoch=0):
        
        def true_solution(x):
            """
            # True Solution (found analytically)

            eq = cos(k*x) - (csc(k) + cot(k)) * sin(k*x)
            """
            k = self.k
            eq = None
            cot_k = np.cos(k) / np.sin(k)
            cosec_k = 1 / np.sin(k)

            if self.p_lb == 1 and self.p_rb == -1:
                eq = np.cos(k * x) - (cosec_k + cot_k) * np.sin(k * x)

            elif self.p_lb == 1 and self.p_rb == 2:
                eq = np.cos(k * x) + (2 * cosec_k - cot_k) * np.sin(k * x)

            return eq

        def nn_approximation(x):
            """
            Neural Network Approximation Solution
            eq = NN(x)
            """

            def create_batch(X):
                output = []
                for d in X:
                    output.append([d])
                return output

            x = create_batch(x)
            x = tf.convert_to_tensor(x, dtype=tf.float32)

            # res = (self.p_lb * (self.x_rb - x)) / (self.x_rb - self.x_lb) + \
            #       (self.p_rb * (x - self.x_lb)) / (self.x_rb - self.x_lb) + \
            #       (x - self.x_lb) * (self.x_rb - x) * self.model(x)
            
            res = self.model(x)

            result = np.zeros(100)
            for i in range(len(res)):
                result[i] = res[i][0]

            return result

        # Generate test data
        test_data = np.linspace(0, 1, 100)

        # Relative error computation
        S = true_solution(test_data)
        DS = nn_approximation(test_data)

        rel_error = np.linalg.norm(DS - S) / np.linalg.norm(S)

        # Plot the output
        figure(1, figsize=(5, 5))
        plt.title(f"Output, f={self.f} Hz\nRelative error  = {rel_error * 100:.3f}%", fontsize=15)
        # plt.ylim([-15, 15])
        plt.xlabel("x", fontsize=15)
        plt.ylabel("p", fontsize=15)    
        plt.plot(test_data, S, label="Original Function")
        plt.plot(test_data, DS, linestyle='dashed', label="Neural Net Approximation")
        
        plt.legend(loc=1, prop={'size': 14})
        # plt.savefig(f'STD_{self.f} Hz-RelativeError.csv', format='csv')

        plt.savefig(f'{self.output_dir}/{self.exp_name}_{epoch}.png')
        plt.show()

        plt.close()

        if epoch > 100 and rel_error * 100 < 0.9:
            print("Optimal relative error reached.")
            return True
        
        return False
