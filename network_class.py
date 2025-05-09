import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initializing random weights with the shape of inputs*neurons
        # in comparison to neurons*inputs, which requires a transpose when multiplying by batched inputs
        # ideally want weights to start small -0.1 to 0.1
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        # for back propagation we need to remember what the inputs were
        self.input = inputs
    def backward(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)
        # store inputs for back propagation
        self.input = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # zero the gradient where input to ReLU was negative
        self.dinputs[self.input <= 0] = 0
    
        
class Activation_Softmax:
    def forward(self, inputs):
        # subtract the max of each input set to prevent overflow when exponentiating by limiting max of exp to 1
        # this does not affect output after normalization
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize the outputs to get probability distribution
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        # Jacobian is computed on a sample-wise basis, so index through the dvalues and output arrays to get individual sample dvalue/output
        # output and dvalues are both shape (#samples x #outputneurons)
        # dinputs will also be (#samples x #outputneurons)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # dot product the jacobian and dvalues for the particular sample, resulting in a vector of length #outputneurons
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        
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
    # takes an array of the probability distributions (dvalues) and array of true values
    # dvalues will be shape (# samples x # ouputs), while true values can either be discrete values (1 x #samples) 
    # or one-hot encoded vectors (#samples x #outputs)
    # to easily do the derivative of the loss function, make sure to transform true values to one-hot encoded form
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # number of output neurons i.e) labels in every sample
        labels = len(dvalues[0])

        # if the true values is in discrete values, turn into one-hot vector
        # ex) if y_true=[0, 2, 1] for 3 samples, equivalent to [[1,0,0],[0,0,1],[0,1,0]]
        # np.eye returns an nxn matrix with 1s on diagonal and 0s everywhere else, n should be # output neurons
        # use the discrete values in y_true to index the np.eye and return the correct one-hot encoded vector
        if (len(y_true.shape)) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        # normalize gradient by dividing by # samples
        self.dinputs = self.dinputs / samples

# combined softmax activation and cross-entropy loss for a faster back propagation
class Activation_Softmax_Loss_CategoricalCrossEntropy():

    # create activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        # pass the inputs of the dense layer through the softmax function
        self.activation.forward(inputs)
        self.output = self.activation.output

        # compute the loss 
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        # if labels are one-hot encoded turn into discrete
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        
        # combined derivative is just the predicted output of softmax minus the y_true
        self.dinputs[range(samples), y_true] -= 1

        # normalize gradient by number of samples
        self.dinputs = self.dinputs / samples
    
class Optimizer_SGD:

    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    # call once before updating parameters in each epoch
    def pre_update_params(self):
        # if the decay is non-zero then update the current learning rate 
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):

        if self.momentum:
            # if the layer doesn't have momentum arrays (i.e first epoch) create them
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            # update layer momentum
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate* layer.dbiases
            layer.bias_momentums = bias_updates
        # for non-momentum updating
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates
        

    # call once after updating parameters in each epoch
    def post_update_params(self):
        self.iterations += 1


class Optimizer_AdaGrad:

    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
    
    # call once before updating parameters in each epoch
    def pre_update_params(self):
        # if the decay is non-zero then update the current learning rate 
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):

        # if layer doesn't have cache array yet make it
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update the weight and bias cache
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        #update the weights and biases
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # call once after updating parameters in each epoch
    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSProp:

    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    # call once before updating parameters in each epoch
    def pre_update_params(self):
        # if the decay is non-zero then update the current learning rate 
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):

        # if layer doesn't have cache array yet make it
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update the weight and bias cache
        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases ** 2
        #update the weights and biases
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # call once after updating parameters in each epoch
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    # call once before updating parameters in each epoch
    def pre_update_params(self):
        # if the decay is non-zero then update the current learning rate 
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):

        # if layer doesn't have cache/momentum array yet make it
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        # update the momentums
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbiases
        # correct the momentums using the beta1
        # we need the "step" parameter to start at 1 for epoch 0
        weight_momentums_corrected = layer.weight_momentums / (1-self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1-self.beta_1 ** (self.iterations + 1))

        # update the weight and bias cache
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases ** 2
        # correct the caches
        weight_cache_corrected = layer.weight_cache / (1-self.beta_2 ** (self.iterations+1))
        bias_cache_corrected = layer.bias_cache / (1-self.beta_2 ** (self.iterations+1))

        #update the weights and biases by dividing the momentums by the square root cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # call once after updating parameters in each epoch
    def post_update_params(self):
        self.iterations += 1
