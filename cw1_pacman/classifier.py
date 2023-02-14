# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class Classifier:
    # Classifier is initialised with nothing passed through. Only initialise 
    # object scope constants and variables here. We may not even use this.
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors = 4)

        pass

    # As far as I can see, we don't have to perform any retraining on the model
    # between sets of games, unless we know the environment changes (I believe
    # it doesn't). Again we may not even use this.
    def reset(self):
        pass
    
    # Trains on training dataset, the arrays of input <data>, on target, or 
    # output singular integer <target>.
    #
    # <data> = [001011101010...]
    # <target> = 0 or 1 or 2 or 3
    #
    # This happens before classifier is run on the environment.
    def fit(self, data, target):
        self.model = MLP(len(data), len(data) + 5, 4)
        prediction = self.model.forward(data)
        print(prediction)
        pass

    # We simply output the integer output/prediction given a singular feature
    # array. Current legal moves are given as array of string 'North', 'East',
    # 'South', 'West', and 'Stop'.
    #
    # ClassifierAgent.py handles output conversion of the number to these 
    # strings taken by the environment. So we may not even use legal at all.
    def predict(self, data, legal=None):
        
        # Currently only outputs 1 (pretty sure that means East). 
        return 1
        
# Multilayer Perception Class
# Will have singular of both forward and backward passes.
# Will perform batch back progagation. 
class MLP:
    def __init__(self, input_size, hidden_size, output_size = 4):
        self.input_size = input_size # unaccounted bias
        self.hidden_size = hidden_size # unaccounted bias
        self.output_size = output_size
        # Input i, output node j.
        # Note: input and hidden sizes are FLIPPED for easier notation during
        # calculation later on.
        self.w_j_i = np.random.uniform(low = -1, high = 1,
                                       size = (self.hidden_size, 
                                               self.input_size + 1))
        
        # Input node j, output k
        # Note: hidden and output sizes are FLIPPED for easier notation during
        # calculation later on.
        self.w_k_j = np.random.uniform(low = -1, high = 1,
                                       size = (self.output_size, 
                                               self.hidden_size + 1))

    # ReLu activation function
    def activation_ReLu(self, x):
        return max(0.0, x)

    # Sigmoid activation function
    def activation_Signmoid(self, x):
        return 1/(1 + np.exp(-x))

    # Initial part of forward pass, iterates over all inputs for the output
    # hidden node j. Includes activation function use.
    def forward_input_j(self, input, j):
        temp = self.w_j_i[j][0] # w_j_0
        for i, x in enumerate(input[0]):
            temp += x * self.w_j_i[j][i + 1]
        return self.activation_ReLu(temp)

    # 2nd part of forward pass, iterates over all y_j for the output k node z_k.
    # Includes activation functions use.
    def forward_hidden_k(self, y, k):
        temp = self.w_k_j[k][0] # w_k_0
        for j, y_j in enumerate(y):
            temp += y_j * self.w_k_j[k][j + 1]
        return self.activation_Signmoid(temp)

    def forward(self, input):
        # z_k = act_func( w_k_0 + sum_over_hidden_nodes( w_k_j * act_func( w_j_0 + sum_over_inputs( w_j_i * x_i ))))
       
        y = np.zeros(self.hidden_size)
        # Calculate all outputs of hidden nodes, y
        for j, y_j in enumerate(y):
            y[j] = self.forward_input_j(input, j)

        z = np.zeros(self.output_size)
        # Calculate all outputs of output nodes, z
        for k, z_k in enumerate(z):
            z_k = self.forward_hidden_k(y, k)
        
        # Return index of the maximum probability selection.
        # Can change to perform a random selection, or any other acceptable 
        # method.
        return np.argmax(z)

        

            
        
            
        
            
            
        
        return 1
    
    