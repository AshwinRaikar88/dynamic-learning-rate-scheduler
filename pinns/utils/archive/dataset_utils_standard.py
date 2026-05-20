"""
Author: Ashwin Raikar
Version: 1.0.2
"""

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure


def generate_data(x_lb, x_rb, nc):
    """
    GENERATE RANDOM INTERNAL COLLOCATION POINTS IN THE DOMAIN

    Generate any random float between x_lb to x_rb
    use round() if you need number to specified decimal places

    param x_lb: x left boundary (int)
    param x_rb: x right boundary (int)
    param nc: Number of Internal collocation points (int)

    """
    data = []
    for i in range(0, nc):
        data.append([round(np.random.uniform(x_lb, x_rb), 5)])

    return data


def plot_data(data, nc):
    cm = matplotlib.colormaps['winter']

    x = np.linspace(0, nc, nc)
    y = data

    plt.xlabel('Number of internal collocation points', fontsize=10)
    plt.ylabel('Values', fontsize=10)
    plt.scatter(x, y, c=y, cmap=cm)
    plt.show()


def create_batches(data, batch_size):
    output = []
    num_items = len(data)
    num_batches = num_items // batch_size

    print(num_batches)

    lb = 0
    rb = batch_size

    for i in range(num_batches):
        output.append(data[lb:rb])
        lb = rb
        rb += batch_size
        if lb == num_items:
            return output

    output.append(data[lb:])
    for i in range(batch_size - len(data[lb:])):
        output[-1].append([0.0])

    return output


def plot_output(f, k, p_l, p_r, NN):
    def true_solution(x):
        """
        # True Solution (found analytically)

        eq = cos(k*x) - (csc(k) + cot(k)) * sin(k*x)
        """

        eq = None
        cot_k = np.cos(k) / np.sin(k)
        cosec_k = 1 / np.sin(k)

        if p_l == 1 and p_r == -1:
            eq = np.cos(k * x) - (cosec_k + cot_k) * np.sin(k * x)

        elif p_l == 1 and p_r == 2:
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

        res = NN.model(x)

        result = np.zeros(100)
        for i in range(len(res)):
            result[i] = res[i][0]

        return result

    # Generate test data
    test_data = np.linspace(0, 1, 100)

    S = true_solution(test_data)
    DS = nn_approximation(test_data)

    norm_S = np.linalg.norm(np.abs(S))
    norm_DS = np.linalg.norm(np.abs(S) - np.abs(DS))
    rel_error = norm_DS / norm_S

    print(rel_error)

    figure(1, figsize=(5, 5))
    plt.title(f"Standard model, f={f} Hz\nRelative error  = {rel_error * 100:.3f}%", fontsize=15)
    # plt.ylim([-15, 15])
    plt.xlabel("x", fontsize=15)
    plt.ylabel("p", fontsize=15)
    plt.plot(test_data, S, label="Original Fucntion")
    plt.plot(test_data, DS, linestyle='dashed', label="Neural Net Approximation")
    plt.legend(loc=1, prop={'size': 14})
    plt.show()


def plot_output_gmo(f, k, p_l, p_r, NN):
    def true_solution(x):
        """
        # True Solution (found analytically)

        eq = cos(k*x) - (csc(k) + cot(k)) * sin(k*x)
        """

        eq = None
        cot_k = np.cos(k) / np.sin(k)
        cosec_k = 1 / np.sin(k)

        if p_l == 1 and p_r == -1:
            eq = np.cos(k * x) - (cosec_k + cot_k) * np.sin(k * x)

        elif p_l == 1 and p_r == 2:
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

        res, _ = NN.model(x)

        result = np.zeros(100)
        for i in range(len(res)):
            result[i] = res[i][0]

        return result

    # Generate test data
    test_data = np.linspace(0, 1, 100)

    S = true_solution(test_data)
    DS = nn_approximation(test_data)

    norm_S = np.linalg.norm(np.abs(S))
    norm_DS = np.linalg.norm(np.abs(S) - np.abs(DS))
    rel_error = norm_DS / norm_S

    print(rel_error)

    figure(1, figsize=(5, 5))
    plt.title(f"Standard model, f={f} Hz\nRelative error  = {rel_error * 100:.3f}%", fontsize=15)
    # plt.ylim([-15, 15])
    plt.xlabel("x", fontsize=15)
    plt.ylabel("p", fontsize=15)
    plt.plot(test_data, S, label="Original Fucntion")
    plt.plot(test_data, DS, linestyle='dashed', label="Neural Net Approximation")
    plt.legend(loc=1, prop={'size': 14})
    plt.show()
