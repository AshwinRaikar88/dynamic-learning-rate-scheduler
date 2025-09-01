from time import time
from utils.dataset_utils import *

from pinn.pinn_acoustic import PINN


if __name__ == '__main__':
    freqencies = [250, 500, 1000, 1500, 2000]

    time_taken = []
    exp_id = '03'
    
    for freq in freqencies:
    # for freq in range(1010, 1031):
        # Variables associated with physics
        f = freq                                # Frequency
        c = 340                                 # Speed of sound
        x_lb = 0                                # Left boundary
        x_rb = 1                                # Right boundary
        p_lb = 1                                # Pressure at the left boundary
        p_rb = -1                               # Pressure at the right boundary
        k = 2 * np.pi * f / c                   # Wave number

        numIntColPoints = 10000
        lr = 0.005
        batch_size = 1000
        num_epochs = 10000

        # Generate random data points
        data = generate_data(x_lb=0, x_rb=1, nc=numIntColPoints)
        # plot_data(data=data, nc=numIntColPoints)

        # DLRS config
        # delta_i = 0.1
        # delta_d = 0.5

        layers = [100, 100, 100]
        # Initialize the PINN model
        # NN = PINN(frequency=f, exp_name=f'trial_solution/dlrs_di{delta_i}_dd{delta_d}_{f}Hz')
        # NN = PINN(exp_name=f'extras/Layers_{len(layers)}_{f}Hz', frequency=f, layers_list=layers)
        # NN = PINN(exp_name=f'extras/Layers_{len(layers)}_exponential_{f}Hz', frequency=f, layers_list=layers)
        # NN = PINN(exp_name=f'extras/VW{exp_id}_{f}Hz', frequency=f, layers_list=layers)
        NN = PINN(exp_name=f'VW{exp_id}_{f}Hz', frequency=f, layers_list=layers)
        
        NN.load_data(data, batch_size)

        # Start training
        t1 = time()
        # NN.train(epochs=num_epochs, learning_rate=lr, optimizer='adam', dlrs=(delta_i, delta_d))
        NN.train(epochs=num_epochs, learning_rate=lr, optimizer='adam', dynamic_lr=['dlrs'])
        # NN.train(epochs=num_epochs, learning_rate=lr, optimizer='adam', dlrs='exponential')
        t2 = time() - t1

        print(f"Total time taken for training {num_epochs} epochs: {t2:.3f} s")
        time_taken.append(f"{f}Hz ={t2:.3f} s")

        # Plot the output
        plot_output(f, k, p_lb, p_rb, NN)
    print(time_taken)
