import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initializing random weights with the shape of inputs*neurons
        # in comparison to neurons*inputs, which requires a transpose when multiplying by batched inputs
        # ideally want weights to start small -0.1 to 0.1
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)
        
class Activation_Softmax:
    def forward(self, inputs):
        # subtract the max of each input set to prevent overflow when exponentiating by limiting max of exp to 1
        # this does not affect output after normalization
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize the outputs to get probability distribution
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
# inherit from base Loss class
class Loss_CategoricalCrossEntropy(Loss):
    # input the predicted outputs and the expected outputs
    def forward(self, y_pred, y_true):
        # inputs in batch
        samples = len(y_pred)
        # clip the values to be within 0 and 1, to prevent log(0)=inf which throws off the mean
        y_pred_clipped = np.clip(y_pred, 1.0e-7, 1.0-1.0e-7)
        
        # handle scalar values for expected output
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # handle one-hot encoded values
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        
        
        
# create input. 100 feature sets i.e) a batch of 100 where each set has 2 inputs, an x and a y position
# x and y are stored as a vector pair in the X 
# while the "y" being returned from spiral_data is actually the category for each input
# 3 classification sets, i.e) 3 outputs
X, y = spiral_data(samples=100, classes=3)
    
# 2 inputs, 3 neurons in hidden layer
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# dense2 is the output layer. 3 neuron inputs from hidden layer, 3 outputs since 3 classes
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)
print(loss)

