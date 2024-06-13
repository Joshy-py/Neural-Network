import numpy as np

# Sigmoid function again
def nonlin(i):
    output = 1/(1 + np.exp(-i))
    return output

# Takes the sigmoid output and turns it into it's derivative
def nonlin_output_to_deriv(output):
    return output * (1 - output)

# Input dataset
x = np.array([[0,1], [0,1], [1,0], [1,0]])

# Output dataset
y = np.array([[0,0,1,1]]).T

# Randomly seeding values (because its good practice)
np.random.seed(1)

# Synapse 1 connecting layer 1 and 2
syn0 = 2 * np.random.random((2,1)) - 1

# Where the magic (iterations) happens
for j in range(10000):

    l0 = x # Layer 1, the inputs
    l1 = nonlin(np.dot(l0,syn0)) # Layer 2, the hidden layer

    l1_error = l1 - y # The layer 2 error, comparing the layer 2 prediction of the output with the ACTUAL output

    l1_delta = l1_error * nonlin_output_to_deriv(l1) # Tweaks the model based on the error that we recieved

    syn0_deriv = np.dot(l0.T, l1_delta)

    syn0 -= syn0_deriv # Changes the weights accordingly

print("Output After Training:")
print(l1)