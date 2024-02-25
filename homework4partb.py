# Chase Whitfield
# MAE 3403 Spring 2024
# Homework 4 Problem B
# 02/26/2024

import numpy as np
from scipy.optimize import fsolve

def equation1(x):
    """
    Defines the first equation: x - 3 * cos(x)
    :param x: (float): The input value for x.
    :return: (float): The value of the function at x.
    """
    return x - 3 * np.cos(x)

def equation2(x):
    """
    Defines the second equation: cos(2*x) * x^3

    :param x: (float): The inout value of x.
    :return: (float): The value of the function at x.
    """
    return np.cos(2 * x) * x**3

# Initial guesses for the roots
initial_guesses = [1, 2]

# Initial guesses are not required, they do help with the algorithm and improve the performance of the code.
# Solve equation 1
root1 = fsolve(equation1, initial_guesses[0])
root2 = fsolve(equation1, initial_guesses[1])

print("Roots of x - 3cos(x) = 0:")
print("Root 1:", root1)
print("Root 2:", root2)

# Solve Equation 2
root3 = fsolve(equation2, initial_guesses[0])
root4 = fsolve(equation2, initial_guesses[1])

print("Roots of cos(2x) * x^3 = 0:")
print("Root 1:", root3)
print("Root 2:", root4)

# Evaluate both functions at the roots found above
intersection1 = equation1(root3)
intersection2 = equation1(root4)

print("Intersection Points:")
print("For cos(2x) * x^3 = 0 at root 1 of x - 3cos(x) = 0", intersection1)
print("For cos(2x) * x^3 =0 at root 2 of x - 3cos(x) = 0:", intersection2)

# Used ChatGPT to use the random guess functions
# Used ChatGPT to help with the intersection functions