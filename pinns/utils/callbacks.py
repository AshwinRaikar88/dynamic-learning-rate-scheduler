import numpy as np


class DLRS:
    def __init__(self, lr_upper_threshold=0.1, lr_lower_threshold=1e-5, delta_increase=0.5, delta_decrease=1):
        self.lr_upper_threshold = lr_upper_threshold
        self.lr_lower_threshold = lr_lower_threshold

        self.del_i = delta_increase
        self.del_d = delta_decrease


    def decrease_ut(self):
        self.lr_upper_threshold -= self.lr_upper_threshold / 2
        return self.lr_upper_threshold


    def step(self, lr, loss_metrics):
        # Get the order of the learning rate
        order_lr = np.floor(np.log10(lr))
        # order_lr = np.floor(np.log10(lr))
        # print(f"Old lr = {lr}, lr order = {order_lr}")

        # Calculate mean of a batch of loss values
        Lm = np.mean(loss_metrics)

        # Calculate the normalized slope - Observations
        Norm = (loss_metrics[-1] - loss_metrics[0]) / Lm
        # print(f"Obs = {Norm} ")

        if Norm > 2:  # this means, if Norm is > 0
            # This means the loss increased so decrease the lr
            Norm *= self.del_d * np.pow(10, order_lr)
        elif Norm > 0:
            Norm *= 1 * np.pow(10, order_lr)
        else:
            # This means the loss is decreasing so gradually increase the lr
            Norm *= self.del_i * np.pow(10, order_lr)

        # update learning rate
        lr -= Norm
        # print(f"Norm = {Norm} \n Learning rate= {lr}")

        if lr < self.lr_lower_threshold:
            lr = self.lr_lower_threshold
            print(f"DLRS [info]: Lower limit reached, lr = {lr}")
            return round(lr, 8), False

        elif lr > self.lr_upper_threshold:
            lr = self.lr_upper_threshold
            print(f"DLRS [info]: Upper limit reached, lr = {lr}")
            return round(lr, 8), True
        else:
            # order_lr = np.floor(np.log10(lr))
            # print(f"DLRS [info]: New lr = {lr}, lr order = {order_lr}")
            return round(lr, 8), True
    

    def step_v2(self, lr, loss_metrics):
        # Get the order of the learning rate
        order_lr = np.floor(np.log10(lr))
        # print(f"Old lr = {lr}, lr order = {order_lr}")

        # Calculate mean of a batch of loss values
        Lm = np.mean(loss_metrics)

        # Calculate the normalized slope - Observations
        Norm = (loss_metrics[-1] - loss_metrics[0]) / Lm
        # print(f"Obs = {Norm} ")

        if Norm >= 1:  # this means, if Norm is > 0
            # Loss increase, Decrease Lr
            Norm *= self.del_d * np.pow(10, order_lr)
        elif Norm < 0:
            # Loss decrease, Increase Lr
            Norm *= self.del_i * np.pow(10, order_lr)
        else:
            # Between 0 to 1 loss is stagnating
            Norm *= np.pow(10, order_lr)

        # update learning rate
        lr -= Norm
        # print(f"Norm = {Norm} \n Learning rate= {lr}")

        if lr < self.lr_lower_threshold:
            lr = self.lr_lower_threshold
            print(f"DLRS [info]: Lower limit reached, lr = {lr}")
            return round(lr, 8), False

        elif lr > self.lr_upper_threshold:
            lr = self.lr_upper_threshold
            print(f"DLRS [info]: Upper limit reached, lr = {lr}")
            return round(lr, 8), True
        else:
            # order_lr = np.floor(np.log10(lr))
            # print(f"DLRS [info]: New lr = {lr}, lr order = {order_lr}")
            return round(lr, 8), True
        

    def step_v3(self, lr, loss_metrics):
        # Get the order of the learning rate
        order_lr = np.floor(np.log10(lr))
        print(f"order lr = {order_lr}")
        
        # Calculate mean of a batch of loss values
        Lm = np.mean(loss_metrics)

        # Calculate the normalized slope - Observations
        Ld = (loss_metrics[-1] - loss_metrics[0]) / Lm

        Norm = (Ld/Lm) * np.pow(10, order_lr)

        # update learning rate
        lr -= Norm
        print(f"Delta L = {Ld}\nLm = {Lm}\nNorm = {Norm}\nLearning rate= {lr}")

        if lr < self.lr_lower_threshold:
            lr = self.lr_lower_threshold
            print(f"DLRS [info]: Lower limit reached, lr = {lr}")
            return round(lr, 8), False

        elif lr > self.lr_upper_threshold:
            lr = self.lr_upper_threshold
            print(f"DLRS [info]: Upper limit reached, lr = {lr}")
            return round(lr, 8), True
        else:
            # order_lr = np.floor(np.log10(lr))
            # print(f"DLRS [info]: New lr = {lr}, lr order = {order_lr}")
            return round(lr, 8), True