# Dynamic Learning Rate Scheduler

This is the Official implementation and experimental results for the paper *“Improving Neural Network Training Using Dynamic Learning Rate Schedule for PINNs and Image Classification”*.

Here is the [link to paper](https://doi.org/10.1016/j.mlwa.2025.100697).


## 📌 Overview
This repository contains code and experiments exploring **dynamic learning rate scheduler** to improve training stability and performance for:

- **Physics-Informed Neural Networks (PINNs)**  
- **Image Classification Models**

The work investigates DLRS adaptive scheduling strategies that respond to training dynamics, aiming to enhance convergence speed, reduce overfitting, and boost accuracy.


## 🚀 Features

- Modular **dynamic learning rate scheduler** implementation
- Example training pipelines for PINNs and image classification tasks
- Pre-configured experiment scripts with reproducible settings
- Easy-to-extend dlrs module for other deep learning tasks


## 🛠️ Installation

```bash
git clone https://github.com/AshwinRaikar88/dynamic-learning-rate-scheduler.git
cd dynamic-learning-rate-scheduler
pip install -r requirements.txt
```

## 📜 License

This project is licensed under the **Apache License 2.0** – see the [LICENSE](./LICENSE) file for details.

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{DHARANALAKOTA2025100697,
  title   = {Improving neural network training using dynamic learning rate schedule for PINNs and image classification},
  author  = {Veerababu Dharanalakota and Ashwin Arvind Raikar and Prasanta Kumar Ghosh},
  journal = {Machine Learning with Applications},
  volume  = {21},
  pages   = {100697},
  year    = {2025},
  issn    = {2666-8270},
  doi     = {https://doi.org/10.1016/j.mlwa.2025.100697},
  url     = {https://www.sciencedirect.com/science/article/pii/S2666827025000805},
  keywords = {Adaptive learning, Multilayer perceptron, CNN, MNIST, CIFAR-10}
}
