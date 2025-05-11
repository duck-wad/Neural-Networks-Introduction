import matplotlib.pyplot as plt
import numpy as np
import nnfs
from nnfs.datasets import sine_data

nnfs.init()

''' DEFINE INPUTS AND HYPERPARAMETERS '''
X, y = sine_data()
plt.plot(X,y)
plt.show()