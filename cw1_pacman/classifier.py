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
        self.w_j_i = np.random.uniform(low = -1, high = 1,
                                       size = (self.input_size + 1, 
                                               self.hidden_size))
        self.w_k_j = np.random.uniform(low = -1, high = 1,
                                       size = (self.hidden_size + 1, 
                                               self.output_size))
    
    def forward(self, input):
        # z_k = act_func( w_k_0 + sum_over_hidden_nodes( w_k_j * act_func( w_j_0 + sum_over_inputs( w_j_i * x_i ))))
        temp = w_j_i[0] # w_j_0
        for i, x in input:
            temp += x*w_j_i[i+1]
        
            
            
        
        return 1
    
    