#Elva Soto
#February 23, 2024

import numpy as np
from scipy.linalg import solve


#left hand side of the first matrix
A1 = np.array([[3, 1, -1],
      [1, 4, 1],
      [2, 1, 2]])
#right hand side of the first matrix
b1 = np.array([2, 12, 10])

#left hand side of the second matrix
A2 = np.array([[1, -10, 2, 4],
      [3, 1, 4, 12],
      [9, 2, 3, 4],
      [-1, 2, 7, 3]])
#right hand side of the second matrix
b2 = np.array([2, 12, 21, 37])

#solve the first matrix
x1 = solve(A1, b1)

#solve the second matrix
x2 = solve(A2, b2)
#noticed x2 had long decimal answers, so I created a second x2 that is rounded to 4 decimal places
x2_rounded = np.round(x2, decimals = 4)

#print the answers for each matrix and the rounded solution for the second problem

print("Solution for Matrix #1:", x1)

print("Solution for Matrix #2:", x2)
print("Rounded Solution for Matrix #2 :", x2_rounded)

