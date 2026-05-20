from time import time
from pinn.pinn_main import PINN
from utils.training_utils import *


if __name__ == '__main__':
    freqencies = [250, 500, 1000, 1500, 2000]
    time_taken = []

    training_cfg, physics_cfg, model_cfg, paths_config = load_configuration(config_file_path="../configs/config_dirichlet.yaml")

    # Initialize the PINN model
    NN = PINN(configs=[training_cfg, physics_cfg, model_cfg, paths_config])

    # Start training
    t1 = time()
    NN.train(epochs=num_epochs,
             learning_rate=lr,
             optimizer='adam',
             dynamic_lr=['dlrs'])

    t2 = time() - t1

    print(f"Total time taken for training {num_epochs} epochs: {t2:.3f} s")
