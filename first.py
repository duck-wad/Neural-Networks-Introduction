import sys
import numpy as np
import matplotlib

# outputs from 3 neurons in previous layer serve as inputs for current neuron
inputs = [1.0, 2.0, 3.0] 

# every input has unique weight associated with it
weights = [0.2,0.8,-0.5]

# every unique neuron has a unique bias
bias = 2.0

# output for each neuron is sum of inputs times weights plus bias
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)