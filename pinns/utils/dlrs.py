import numpy as np

class DLRS:
    def __init__(self, delta_increase=0.1, delta_decrease=1, lr_upper_threshold=0.1, lr_lower_threshold=1e-5):
        self.del_i = delta_increase
        self.del_d = delta_decrease
        
        self.lr_upper_threshold = lr_upper_threshold
        self.lr_lower_threshold = lr_lower_threshold

        self.Norm = 0.0

        self.spike_thresh = 0.5

        self.log_data = {"norm": self.Norm, 
                         "spike_thresh": self.spike_thresh,
                         "del_i": self.del_i, 
                         "del_d": self.del_d,}

    def detect_spikes(self):
        if np.abs(self.Norm) > self.spike_thresh:
            # No Aberrations
            return True
        elif 0 <= np.abs(self.Norm) < self.spike_thresh:
            # Abberations
            return False

    def step(self, lr, loss_metrics):
        # Get the order of the learning rate
        order_lr = np.floor(np.log10(lr))
        
        # Calculate mean of a batch of loss values
        Lm = np.mean(loss_metrics)

        # Calculate the normalized slope - Observations
        self.Norm = (loss_metrics[-1] - loss_metrics[0]) / Lm
        
        spikes = self.detect_spikes()

        if self.Norm > 0:
            if spikes:
                self.del_d += self.Norm

            # Loss increase, Decrease Lr
            self.Norm *= self.del_d * np.pow(10, order_lr)

        else:
            if spikes:
                # Decrease del_i: (Since Norm is -ve add it for efficieny) 
                self.del_i += self.Norm
            # Loss decrease, Increase Lr
            self.Norm *= self.del_i * np.pow(10, order_lr)

        # Update learning rate
        lr -= self.Norm

        self.log_data.update({"norm": self.Norm,
                              "del_i": self.del_i, 
                              "del_d": self.del_d})
        
        if lr < self.lr_lower_threshold:
            lr = self.lr_lower_threshold
            print(f"DLRS [info]: Lower limit reached, lr = {lr}")
            return round(lr, 8), self.log_data, False

        elif lr > self.lr_upper_threshold:
            lr = self.lr_upper_threshold
            print(f"DLRS [info]: Upper limit reached, lr = {lr}")
            return round(lr, 8), self.log_data, True
        else:
            return round(lr, 8), self.log_data, True