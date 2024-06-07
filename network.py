import numpy as np

# Sigmoid function --A function that can be used to predict simple 0 and 1 outcomes--
def nonlin(i, deriv = False):
    if (deriv is True):
        return i * (1 - i)
    return 1/(1 + np.exp(-i))

# Input dataset
x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])

# Output dataset
y = np.array([[0,1,1,0]]).T

# Seeds random numbers to ensure calculations are deterministic
np.random.seed(1)

# Randomly weight the values with a mean weight of 0
syn0 = 2 * np.random.random((3,4)) - 1
syn1 = 2 * np.random.random((4,1)) - 1

# Iterates through
for iter in range(60000):

    l0 = x # Layer 1 of the network, contains the input data
    l1 = nonlin(np.dot(l0,syn0)) # Layer 2 of the network, the 'hidden' layer
    l2 = nonlin(np.dot(l1,syn1))

    l2_error = y - l2 # ???

    l1_delta = l2_error * nonlin(l1,True) # ???

    syn0 += np.dot(l0.T,l1_delta) # Update the weights appropriately

print("Output After Traning:")
print(l1)