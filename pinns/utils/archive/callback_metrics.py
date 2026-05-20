import datetime
import tensorflow as tf


class CallbackMetrics(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()

        expName = 'default'
        self.prev_loss = 0
        self.best_loss = 11000

        self.decay_rate = 0.001

        self.lr = None
        self.thresh = 5e-05

        self.count = 0
        self.loss_metrics = []

        current_time = datetime.datetime.now().strftime("%d-%m-%YY_%H-%M-%S")
        train_log_dir = f'logs/new_tape/{expName}_{current_time}/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def update_lr(self, epoch):
        if self.lr > self.thresh:
            self.count += 1
            # self.lr = self.lr / (1+ self.decay_rate * epoch)
            self.lr = self.lr / (1 + self.decay_rate * 500 * self.count)
            self.model.optimizer.learning_rate.assign(self.lr)
            print(f"\nUpdated Learning Rate: {self.lr}")

    def update_lr_finetune(self, epochs):
        self.lr = self.lr / (1 + self.decay_rate * epochs)
        self.model.optimizer.learning_rate.assign(self.lr)
        print(f"\nUpdated Learning Rate: {self.lr}")

    def on_epoch_begin(self, epoch, logs=None):
        self.loss_metrics = []
        if epoch == 0:
            self.lr = self.model.optimizer.lr.numpy()

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        m = tf.keras.metrics.Mean()
        m.update_state(self.loss_metrics)
        loss = m.result().numpy()
        # print("Callback: Loss = ", loss)

        with self.train_summary_writer.as_default():
            tf.summary.scalar(name="loss", data=loss, step=epoch)
        self.train_summary_writer.flush()
        m.reset_state()

        # Custom Early stopping
        if loss > 100 * self.prev_loss and epoch > 100:
            self.model.stop_training = True
        else:
            self.prev_loss = loss

        if loss > 2:
            if epoch % 400 == 0 and epoch != 0:
                self.update_lr(epoch)
        elif loss < 2:
            if epoch % 100 == 0 and epoch != 0:
                self.update_lr_finetune(epoch)
            print(f"\nUpdated Learning Rate: {self.lr}")

        # if 1000 < loss:
        #    self.lr = 0.009
        #    self.model.optimizer.learning_rate.assign(self.lr)
        # elif 100 < loss < 1000:
        #   self.lr = 0.0005
        #   self.model.optimizer.learning_rate.assign(self.lr)
        #   print(f", lr: {self.lr}")
        # elif epoch % 50 == 0 and loss < 100:
        #   self.update_lr(epoch)
        #   print(f"\nUpdated Learning Rate: {self.lr}")

        # Save best model
        # if loss < self.best_loss:
        #   print("\nSaving model: ./weights/best_700")

        #   # self.model.save("./weights/best_700")
        #   self.model.m_save(path="./weights/best_700")
        #   self.best_loss = loss
        #   self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        self.loss_metrics.append(logs[keys[0]])
