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

    return {'test_data': test_data,
            'ground_truth_real': S_r,
            'nn_output_real': DS_r,
            'ground_truth_imaginary': S_i,
            'nn_output_imaginary': DS_i,
            'relative_error_real': rel_error_r,
            'relative_error_imaginary': rel_error_i}
