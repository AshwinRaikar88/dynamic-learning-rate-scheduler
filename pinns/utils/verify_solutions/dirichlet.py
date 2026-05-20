"""
Author: Ashwin Raikar
Version: 1.0.0
Description: Program to verify True solution (found analytically) with BVP solver using
scikit.integrate.solve_bvp for Dirichlet boundary conditions.
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


def helmholtz_equation(x, p):
    """
    p(x) = p[0]
    dp/dx = p[1]
    """
    dp_dx = p[1]
    d2p_dx2 = -k ** 2 * p[0]
    return np.vstack((dp_dx, d2p_dx2))


def boundary_conditions(pa, pb):
    """
    pa - value of solution at left boundary
    pb - value of solution at right boundary

    Dirichlet Boundary conditions
    pa = p_left
    pb = p_right
    """
    return np.array([pa[0] - p_left, pb[0] - p_right])


def true_solution(x, k):
    """
    True solution - found analytically
    """
    cot_k = np.cos(k) / np.sin(k)
    cosec_k = 1 / np.sin(k)

    y_exact = np.cos(k * x) - (cosec_k + cot_k) * np.sin(k * x)
    return y_exact


if __name__ == "__main__":
    frequencies = [100, 500, 750, 1000]  # frequency in Hz

    for f in frequencies:
        c0 = 340  # speed of sound in m/s
        k = 2 * np.pi * f / c0  # wave number

        # Define the domain
        Nx = 100  # Number of grid points
        x_start, x_end = 0.0, 1.0
        x_vals = np.linspace(x_start, x_end, Nx)

        # Define the boundary values
        p_left = 1.0
        p_right = -1.0

        # Define the initial guess for the solution
        p_guess = np.zeros((2, Nx))
        p_guess[0] = np.sin(np.pi * x_vals)  # Initial guess for p(x)

        # Solve the boundary value problem
        solution = solve_bvp(helmholtz_equation, boundary_conditions, x_vals, p_guess)

        # Extract the solution
        x_vals = solution.x
        p_vals = solution.y[0]

        # Plot the solution
        plt.plot(x_vals, true_solution(x_vals, k), label="Exact")
        plt.plot(x_vals, p_vals, label="BVP Solver", linestyle='dashed')
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.title(f'Solution of the Helmholtz equation\n f={f}Hz ')
        plt.legend()
        plt.show()
