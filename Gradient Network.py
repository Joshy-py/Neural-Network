import numpy as np

def nonlin(i):
    output = 1/(1 + np.exp(-i))
    return output

def nonlin_output_to_deriv(output):
    return output * (1 - output)

x = np.array([[0,1], [0,1], [1,0], [1,0]])

y = np.array([[0,0,1,1]]).T

np.random.seed(1)

syn0 = 2 * np.random.random((2,1)) - 1

for j in range(10000):

    l0 = x
    l1 = nonlin(np.dot(l0,syn0))

    l1_error = l1 - y

    l1_delta = l1_error * nonlin_output_to_deriv(l1)

    syn0_deriv = np.dot(l0.T, l1_delta)

    syn0 -= syn0_deriv

print("Output After Training:")
print(l1)