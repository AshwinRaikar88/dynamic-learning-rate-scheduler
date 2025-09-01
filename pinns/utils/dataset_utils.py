"""
Author: Ashwin Raikar
Version: 1.0.4
"""

import numpy as np
import tensorflow as tf
import logging
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

# Set the logging level to suppress warnings
tf.get_logger().setLevel(logging.ERROR)  # or logging.CRITICAL

# After running your code, you can reset the logging level to its default
# tf.get_logger().setLevel(logging.NOTSET)

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


def generate_data_2d(x_lb, x_rb, y_lb, y_rb, nc):
    """
    GENERATE RANDOM INTERNAL COLLOCATION POINTS IN THE DOMAIN

    Generate any random float between x_lb to x_rb, y_lb to y_rb
    use round() if you need number to specified decimal places

    param x_lb: x left boundary (int)
    param x_rb: x right boundary (int)
    param y_lb: y left boundary (int)
    param y_rb: y right boundary (int)
    param nc: Number of Internal collocation points (int)

    """
    data = []
    for i in range(0, nc):
        # data.append((round(np.random.uniform(x_lb, x_rb), 5), round(np.random.uniform(y_lb, y_rb), 5)))
        data.append((np.random.uniform(x_lb, x_rb), np.random.uniform(y_lb, y_rb)))

    # x = np.linspace(-1, 1, nc, dtype='float32')
    # y = np.linspace(-1, 1, nc, dtype='float32')
    # data = np.transpose(np.array(np.meshgrid(x, y)).reshape(2, -1))
  
    return data


def plot_data(data, nc):
    x = None
    y = None
    if len(data[0]) == 1:
        cm = matplotlib.colormaps['winter']

        x = np.linspace(0, nc, nc)
        y = data

        plt.xlabel('Number of internal collocation points', fontsize=10)
        plt.ylabel('Values', fontsize=10)
        plt.scatter(x, y, c=y, cmap=cm)
        plt.show()

    elif len(data[0]) == 2:
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        x = data[:, 0]
        y = data[:, 1]
        cm = matplotlib.colormaps['winter']

        plt.xlabel('X values', fontsize=10)
        plt.ylabel('Y values', fontsize=10)
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

    # Relative error computation
    S = true_solution(test_data)
    DS = nn_approximation(test_data)

    rel_error = np.linalg.norm(DS - S) / np.linalg.norm(S)

    # Plot the output
    figure(1, figsize=(5, 5))
    plt.title(f"Output, f={f} Hz\nRelative error  = {rel_error * 100:.3f}%", fontsize=15)
    # plt.ylim([-15, 15])
    plt.xlabel("x", fontsize=15)
    plt.ylabel("p", fontsize=15)    
    plt.plot(test_data, S, label="Original Function")
    plt.plot(test_data, DS, linestyle='dashed', label="Neural Net Approximation")
    
    plt.legend(loc=1, prop={'size': 14})
    # plt.savefig(f'STD_{f} Hz-RelativeError.csv', format='csv')
    plt.show()
    plt.close()
    # plt.savefig(f'DLRS_{f} Hz-RelativeError.csv', format='csv')
    return [test_data, S, DS]


def plot_output_2D(NN, epoch='', exp_name='', saveplt=False):
    # Generate test data
    nc = 20
    x = np.linspace(-1, 1, nc, dtype='float32')
    y = np.linspace(-1, 1, nc, dtype='float32')
    
    data = np.transpose(np.array(np.meshgrid(x, y)).reshape(2, -1))
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    
    X, Y = np.meshgrid(x, y)
    # Z = NN.model(x=tf.expand_dims(data[:, 0:1], axis=1), y=tf.expand_dims(data[:, 1:2], axis=1)).numpy()
    Z = NN.model(x=data[:, 0:1], y=data[:, 1:2]).numpy()
    Z = Z.reshape((nc, nc))

    # Plot the surface Top view.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=False)

    # Label axis
    ax.set_xlabel('x-axis', linespacing=4)
    ax.set_ylabel('y-axis', linespacing=4)
    ax.set_zlabel('Pressure values', linespacing=4)

    # Customize the z axis.
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(90, -90)
    plt.show()

    # Plot the surface.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap='jet',
                           linewidth=0, antialiased=False)

    # Label axis
    ax.set_xlabel('x-axis', linespacing=4)
    ax.set_ylabel('y-axis', linespacing=4)
    ax.set_zlabel('Pressure values', linespacing=4)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    if saveplt:
        plt.savefig(f'./outputs/PINN2D/{exp_name}-{epoch}.png')
        print(f"Saved images at ./outputs/PINN2D/{exp_name}-{epoch}.png")
        

    plt.show()
    plt.close()

def plot_output_robin(f, k, x1, x2, p_l, h, c1, NN):
    def true_solution(x):
        """
        True Solution (found analytically)
        """
        m = np.cos(k * x1)
        n = np.sin(k * x1)
        o = (-k * np.sin(k * x2)) + (c1 * np.cos(k * x2))
        q = (k * np.cos(k * x2)) + (c1 * np.sin(k * x2))

        B = ((o * p_l) - (m * h)) / ((o * n) - (q * m))
        A = (h - (B * q)) / o

        y_exact = (A * np.cos(k * x)) + (B * np.sin(k * x))
        return y_exact

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

        preds = NN.model(x)
        res = x * preds + x * (1 - x) * (NN.psi(x2) + ((1 - x) + c1 * x) * preds + c1 * p_l - h) + p_l

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
    plt.title(f"Output, f={f} Hz\nRelative error  = {rel_error * 100:.3f}%", fontsize=15)
    # plt.ylim([-15, 15])
    plt.xlabel("x", fontsize=15)
    plt.ylabel("p", fontsize=15)
    plt.plot(test_data, S, label="Original function")
    plt.plot(test_data, DS, linestyle='dashed', label="Neural Net Approximation")

    plt.legend(loc=1, prop={'size': 14})
    # plt.savefig(f'STD_{f} Hz-RelativeError.csv', format='csv')
    plt.show()
    plt.close()
    # plt.savefig(f'DLRS_{f} Hz-RelativeError.csv', format='csv')
    return [test_data, S, DS]


def plot_output_complex(f, k, p_l, p_r, NN):

    def true_solution(x):
        """
        # True Solution (found analytically)

        eq = cos(k*x) - (csc(k) + cot(k)) * sin(k*x)
        """

        eq = None
        cot_k = np.cos(k) / np.sin(k)
        cosec_k = 1 / np.sin(k)

        if p_l == 1.0 and p_r == -1.0:
            eq = np.cos(k * x) - (cosec_k + cot_k) * np.sin(k * x)

        elif p_l == 1 and p_r == 2:
            eq = np.cos(k * x) + (2 * cosec_k - cot_k) * np.sin(k * x)
            
        elif np.imag(p_r) == -0.5:
            # Complex Medium
            eq = np.cos(k * x) + ((0.1-0.5j) * cosec_k - cot_k) * np.sin(k * x)

        eq_r = np.real(eq)
        eq_i = np.imag(eq)

        return eq_r, eq_i

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

        res_r, res_i = NN.model(x)

        result_r = np.zeros(100)
        result_i = np.zeros(100)

        for i in range(len(res_r)):
            result_r[i] = res_r[i][0]
            result_i[i] = res_i[i][0]

        return result_r, result_i

    # Generate test data
    test_data = np.linspace(0, 1, 100)

    S_r, S_i = true_solution(test_data)
    DS_r, DS_i = nn_approximation(test_data)

    rel_error_r = np.linalg.norm(DS_r - S_r) / np.linalg.norm(S_r)
    rel_error_i = np.linalg.norm(DS_i - S_i) / np.linalg.norm(S_i)

    figure(1, figsize=(5, 5))
    plt.title(f"Complex model, f={f} Hz\nRelative error  = {rel_error_r * 100:.3f}%", fontsize=15)
    plt.ylim([-2, 2])
    plt.xlabel("x", fontsize=15)
    plt.ylabel("p", fontsize=15)
    # plt.plot(test_data, S, label="Original Fucntion")

    plt.plot(test_data, S_r, label="True Solution - Real")
    plt.plot(test_data, DS_r, linestyle='dashed', label="Neural Net Approximation - Real")
    plt.legend(loc=1, prop={'size': 14})
    plt.show()

    figure(2, figsize=(5, 5))
    plt.title(f"Complex model, f={f} Hz\nRelative error  = {rel_error_i * 100:.3f}%", fontsize=15)
    # plt.ylim([-0.5, 0.5])
    plt.ylim([-3, 3])
    
    plt.xlabel("x", fontsize=15)
    plt.ylabel("p", fontsize=15)
    # plt.plot(test_data, S, label="Original Fucntion")

    plt.plot(test_data, S_i, label="True Solution - Imag")
    plt.plot(test_data, DS_i, linestyle='dashed', label="Neural Net Approximation - Imag")
    plt.legend(loc=1, prop={'size': 14})
    plt.show()

    plt.close()

    return {'test_data': test_data,
            'ground_truth_real': S_r,
            'nn_output_real': DS_r,
            'ground_truth_imaginary': S_i,
            'nn_output_imaginary': DS_i,
            'relative_error_real': rel_error_r,
            'relative_error_imaginary': rel_error_i}


def plot_output_multi(NN):
    def true_solution(x):
        """
        True Solution (found analytically)

        eq = cos(k*x) - (csc(k) + cot(k)) * sin(k*x)
        """

        p_lb = NN.p_lb 
        p_rb = NN.p_rb 
        
        k1 = NN.k_a 
        k2 = NN.k_b 
        k3 = NN.k_c 
        k4 = NN.k_d 

        eq_a = None
        eq_b = None
        eq_c = None
        eq_d = None

        cot_k1 = np.cos(k1) / np.sin(k1)
        cot_k2 = np.cos(k2) / np.sin(k2)
        cot_k3 = np.cos(k3) / np.sin(k3)
        cot_k4 = np.cos(k4) / np.sin(k4)

        cosec_k1 = 1 / np.sin(k1)
        cosec_k2 = 1 / np.sin(k2)
        cosec_k3 = 1 / np.sin(k3)
        cosec_k4 = 1 / np.sin(k4)

        if p_lb== 1.0 and p_rb == -1.0:
            eq_a = np.cos(k1 * x) - (cosec_k1 + cot_k1) * np.sin(k1 * x)
            eq_b = np.cos(k2 * x) - (cosec_k2 + cot_k2) * np.sin(k2 * x)
            eq_c = np.cos(k3 * x) - (cosec_k3 + cot_k3) * np.sin(k3 * x)
            eq_d = np.cos(k4 * x) - (cosec_k4 + cot_k4) * np.sin(k4 * x)

        # elif p_l == 1 and p_r == 2:
        #     eq = np.cos(k1 * x) + (2 * cosec_k - cot_k) * np.sin(k1 * x)

        return eq_a, eq_b, eq_c, eq_d

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

        res_a, res_b, res_c, res_d, = NN.model(x)

        result_a = np.zeros(100)
        result_b = np.zeros(100)
        result_c = np.zeros(100)
        result_d = np.zeros(100)

        for i in range(len(res_a)):
            result_a[i] = res_a[i][0]
            result_b[i] = res_b[i][0]
            result_c[i] = res_c[i][0]
            result_d[i] = res_d[i][0]

        return result_a, result_b, result_c, result_d

    f = NN.f

    # Generate test data
    test_data = np.linspace(0, 1, 100)

    # Relative error computation
    S_a, S_b, S_c, S_d = true_solution(test_data)
    
    DS_a, DS_b, DS_c, DS_d,  = nn_approximation(test_data)

    # Relative error: A
    rel_error_a = np.linalg.norm(DS_a - S_a) / np.linalg.norm(S_a)

    # Relative error: B
    rel_error_b = np.linalg.norm(DS_b - S_b) / np.linalg.norm(S_b)

    # Relative error: C
    rel_error_c = np.linalg.norm(DS_c - S_c) / np.linalg.norm(S_c)

    # Relative error: D
    rel_error_d = np.linalg.norm(DS_d - S_d) / np.linalg.norm(S_d)

    # Plot the output
    # A
    figure(1, figsize=(5, 5))
    plt.title(f"f={f[0]} Hz\nRelative error  = {rel_error_a * 100:.3f}%", fontsize=15)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("p", fontsize=15)

    # plt.ylim([-2, 2])

    plt.plot(test_data, S_a, label="True Solution")
    plt.plot(test_data, DS_a, linestyle='dashed', label="Neural Net Approximation")

    plt.legend(loc=1, prop={'size': 14})
    plt.show()

    # B
    figure(2, figsize=(5, 5))
    plt.title(f"f={f[1]} Hz\nRelative error  = {rel_error_b * 100:.3f}%", fontsize=15)
    # plt.ylim([-3, 3])

    plt.xlabel("x", fontsize=15)
    plt.ylabel("p", fontsize=15)

    plt.plot(test_data, S_b, label="True Solution")
    plt.plot(test_data, DS_b, linestyle='dashed', label="Neural Net Approximation")

    plt.legend(loc=1, prop={'size': 14})
    plt.show()

    # C
    figure(2, figsize=(5, 5))
    plt.title(f"f={f[2]} Hz\nRelative error  = {rel_error_c * 100:.3f}%", fontsize=15)
    # plt.ylim([-3, 3])

    plt.xlabel("x", fontsize=15)
    plt.ylabel("p", fontsize=15)

    plt.plot(test_data, S_c, label="True Solution")
    plt.plot(test_data, DS_c, linestyle='dashed', label="Neural Net Approximation")

    plt.legend(loc=1, prop={'size': 14})
    plt.show()

    # D
    figure(2, figsize=(5, 5))
    plt.title(f"f={f[3]} Hz\nRelative error  = {rel_error_d * 100:.3f}%", fontsize=15)
    # plt.ylim([-3, 3])

    plt.xlabel("x", fontsize=15)
    plt.ylabel("p", fontsize=15)

    plt.plot(test_data, S_d, label="True Solution")
    plt.plot(test_data, DS_d, linestyle='dashed', label="Neural Net Approximation")

    plt.legend(loc=1, prop={'size': 14})
    plt.show()

    plt.close()
