import numpy as np

def nonlin(i):
    output = 1/(1 + np.exp(-i))
    return output

def nonlin_output_to_deriv(output):
    return output * (1 - output)

x = np.array([[0,1], [0,1], [1,0], [1,0]])

y = np.array([[0,0,1,1]]).T

np.random.seed(1)

syn0 = 2 * np.random.random(2,1) - 1