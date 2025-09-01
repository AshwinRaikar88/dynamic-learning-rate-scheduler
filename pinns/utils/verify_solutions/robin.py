"""
Author: Ashwin Raikar
Version: 1.0.0
Description: Program to verify True solution (found analytically) with BVP solver using
scikit.integrate.solve_bvp for Robin boundary conditions.
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

    Robin Boundary conditions
    pa = p1
    dpb/dx + cpb = h
    """

    return np.array([pa[0] - p_left, pb[1] + c1 * pb[0] - h])


def true_solution(x, k):
    """
    # True Solution (found analytically)

    """
    m = np.cos(k * x1)
    n = np.sin(k * x1)
    o = (-k * np.sin(k * x2)) + (c1 * np.cos(k * x2))
    q = (k * np.cos(k * x2)) + (c1 * np.sin(k * x2))

    B = ((o * p_left) - (m * h)) / ((o * n) - (q * m))
    A = (h - (B * q)) / o

    y_exact = (A * np.cos(k * x)) + (B * np.sin(k * x))
    return y_exact


frequencies = [100, 500, 750, 1000, 1500, 2000]  # frequency in Hz

for f in frequencies:
    c0 = 340  # speed of sound in m/s
    k = 2 * np.pi * f / c0  # wave number

    # Define the domain
    Nx = 100  # Number of grid points
    x1, x2 = 0.0, 1.0
    x_vals = np.linspace(x1, x2, Nx)

    # Define the boundary values
    p_left = 1.0
    h = 5
    c1 = 2

    # Define the initial guess for the solution
    p_guess = np.zeros((2, Nx))
    p_guess[0] = np.sin(np.pi * x_vals)  # Initial guess for u(x) - this is a rough guess of the nature of the solution

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
