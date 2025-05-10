import nnfs
from nnfs.datasets import spiral_data
from network_class import *

nnfs.init()

''' SPECIFY THE INPUTS AND HYPERPARAMETERS '''
# create input. 100 feature sets i.e) a batch of 100 where each set has 2 inputs, an x and a y position
# x and y are stored as a vector pair in the X 
# while the "y" being returned from spiral_data is the category for each input
# 3 classification sets, i.e) 3 outputs
X, y = spiral_data(samples=100, classes=3)
epochs = 10001
num_inputs = 2
num_neurons = 64
num_outputs = 3

learning_rate_hp = 0.05
momentum_hp = 0.9
decay_hp = 5e-7
epsilon_hp = 1e-7
rho_hp = 0.999
beta_1_hp = 0.9
beta_2_hp = 0.999


   
''' INITIALIZE THE NETWORK '''
# 2 inputs, 64 neurons in hidden layer
dense1 = Layer_Dense(num_inputs, num_neurons)
activation1 = Activation_ReLU()

# dense2 is the output layer. 3 outputs since 3 classes
dense2 = Layer_Dense(num_neurons, num_outputs)
# create the combined softmax and loss function object 
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
# create the SGD optimizer
#optimizer = Optimizer_SGD(learning_rate=learning_rate_hp, decay=decay_hp, momentum=momentum_hp)
# create the AdaGrad optimizer
#optimizer = Optimizer_AdaGrad(learning_rate=learning_rate_hp, decay=decay_hp, epsilon=epsilon_hp)
# create the RMSProp optimizer
#optimizer = Optimizer_RMSProp(learning_rate=learning_rate_hp, decay=decay_hp, epsilon=epsilon_hp, rho=rho_hp)
# create the Adam optimizer
optimizer = Optimizer_Adam(learning_rate=learning_rate_hp, decay=decay_hp, 
                           epsilon=epsilon_hp, beta_1=beta_1_hp, beta_2=beta_2_hp)



''' RUN THE TRAINING LOOP'''
for epoch in range(epochs):
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    #activation2.forward(dense2.output)
    loss = loss_activation.forward(dense2.output, y)

    # calculate the accuracy of output to targets
    predictions = np.argmax(loss_activation.output, axis=1)
    # handle one-hot encoded output and convert to sparse
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        # print the loss
        print("Epoch: ", epoch, " Accuracy: ", round(accuracy, 5), 
              " Loss: ", round(loss,5), " LR: ", round(optimizer.current_learning_rate,5))

    # using the combined softmax / loss class for faster backprop
    # compute the gradients of the layer 2softmax inputs
    loss_activation.backward(loss_activation.output, y)
    # compute the gradients for layer 2 weights and biases, as well as gradients of layer 1 ReLU outputs
    dense2.backward(loss_activation.dinputs)
    # compute the gradients for the layer 1 ReLU inputs
    activation1.backward(dense2.dinputs)
    # compute the gradients for layer 1 weights and biases
    dense1.backward(activation1.dinputs)

    # update the learning rate 
    optimizer.pre_update_params()
    # update the weights and biases from the computed gradients
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    # update iterations
    optimizer.post_update_params()



''' VALIDATE DATA WITH FORWARD PASS '''
X_test, y_test = spiral_data(samples=100, classes=3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
# handle one-hot encoded output and convert to sparse
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions==y_test)

print(" Accuracy: ", round(accuracy, 5), " Loss: ", round(loss,5))
