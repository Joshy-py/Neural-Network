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
syn0 = 2 * np.random.random((3,4)) - 1 # Synapse connecting and weighting l0 and l1 values
syn1 = 2 * np.random.random((4,1)) - 1 # Synapse connecting and weighting l1 and l2 values

# Iterates through
for iter in range(60000):

    l0 = x # Layer 1 of the network, contains the input data
    l1 = nonlin(np.dot(l0,syn0)) # Layer 2 of the network, the 'hidden' layer
    l2 = nonlin(np.dot(l1,syn1)) # Layer 3 of the network, acts as the hypothesis and approximates the correct answer over time

    l2_error = y - l2 # ???

    if (iter % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * nonlin(l2,deriv=True) # Determines how much to tweak the network based on how close & confident it was to the true value

    l1_error = l2_delta.dot(syn1.T) # Determines how much each Layer 2 value changed the actual output

    l1_delta = l1_error * nonlin(l1,deriv=True) # Tweaks the layer 2 values based on confidence & direction

    # Update the weights appropriately
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta) 

print("Output After Traning:")
print(l2)