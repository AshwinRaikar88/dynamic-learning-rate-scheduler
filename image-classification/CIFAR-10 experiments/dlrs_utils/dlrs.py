import numpy as np

class DLRS:
    def __init__(self, lr_upper_threshold=0.1, lr_lower_threshold=1e-7, delta_increase=0.5, delta_decrease=1):
        self.lr_upper_threshold = lr_upper_threshold
        self.lr_lower_threshold = lr_lower_threshold

        self.del_i = delta_increase
        self.del_d = delta_decrease

    def decrease_ut(self):
        self.lr_upper_threshold -= self.lr_upper_threshold/2
        return self.lr_upper_threshold

    def step(self, lr, loss_metrics):
        # Get the order of the learning rate
        order_lr = np.math.floor(np.math.log10(lr))  
        # print(f"Old lr = {lr}, lr order = {order_lr}")
        
        # Calculate mean of a batch of loss values
        Lm = np.mean(loss_metrics)

        # Calculate the normalized slope - Observations
        Norm = (loss_metrics[-1] - loss_metrics[0]) / Lm
        # print(f"Obs = {Norm} ")

        if Norm == 0:
            Norm += 0.1e-8 

        if Norm > 2:
            # This means the loss increased so decrease the lr          
            Norm *= self.del_d * np.math.pow(10, order_lr)
        elif Norm > 1e-8: # this means, if Norm is > 0
            Norm *= -2 * np.math.pow(10, order_lr)
        else:
            # This means the loss is decreasing so gradually increase the lr
            Norm *= self.del_i * np.math.pow(10, order_lr)

        # update learning rate
        lr -= Norm
        lr = abs(lr)
        # print(f"Norm = {Norm} \n Learning rate= {lr}")

        if lr < self.lr_lower_threshold:
            lr = self.lr_lower_threshold
            # print(f"DLRS [info]: Lower limit reached, lr = {lr}")

        elif lr > self.lr_upper_threshold:
            lr = self.lr_upper_threshold
            # print(f"DLRS [info]: Upper limit reached, lr = {lr}")
        
        # order_lr = np.math.floor(np.math.log10(lr))
        # print(f"DLRS [info]: New lr = {lr}, lr order = {order_lr}")
        return lr