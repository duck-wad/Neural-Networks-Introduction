import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import sine_data
from network_class import *


nnfs.init()



''' DEFINE INPUTS AND HYPERPARAMETERS '''
X, y = sine_data()

epochs = 10001
num_inputs = 1
num_neurons = 64
num_outputs = 1

learning_rate_hp = 0.0005
momentum_hp = 0.9
decay_hp = 1e-3
epsilon_hp = 1e-7
rho_hp = 0.999
beta_1_hp = 0.9
beta_2_hp = 0.999
L1_weight_hp = 0.0
L1_bias_hp = 0.0
L2_weight_hp = 5e-4
L2_bias_hp = 5e-4
dropout_rate_hp = 0.1
std_dev_fraction = 250.0



''' DEFINE THE NETWORK ARCHITECTURE '''

# network of 1 input layer, 2 hidden layer, 1 output layer
# relu function requires more than 1 hidden layer to map nonlinear function
dense1 = Layer_Dense(num_inputs, num_neurons)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(num_neurons, num_neurons)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(num_neurons, num_outputs)
activation3 = Activation_Linear()
loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_Adam(learning_rate=learning_rate_hp, decay=decay_hp)

accuracy_limit = np.std(y) / std_dev_fraction



''' DEFINE THE TRAINING LOOP '''

for epoch in range(epochs):
    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    # calculate the data and regularization loss (if using regularization)
    data_loss = loss_function.calculate(activation3.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2) + loss_function.regularization_loss(dense3)

    # overall loss is sum of data and regularization loss
    loss = data_loss + regularization_loss

    # calculate the accuracy of the model
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_limit)

    if not epoch % 100:
        # print current performance data
        print("Epoch: ", epoch, " Accuracy: ", round(accuracy, 5), 
              " Loss: ", round(loss,5), 
              " Data loss: ", round(data_loss, 5),
              " Regularization loss: ", round(regularization_loss, 5),
              " LR: ", round(optimizer.current_learning_rate,5))
        
    # backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update the weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()



''' TEST PERFORMANCE '''

X_test, y_test = sine_data()
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

# measure performance
data_loss = loss_function.calculate(activation3.output, y_test)
predictions = activation3.output
accuracy = np.mean(np.absolute(predictions - y) < accuracy_limit)
print("Validation performance:", " Accuracy: ", round(accuracy, 5), " Data loss: ", round(loss,5))


fig, ax = plt.subplots(1, 1)

ax.plot(X_test, y_test, label="Test data")
ax.plot(X_test, activation3.output, label="Predicted output")
ax.legend()
plt.show()