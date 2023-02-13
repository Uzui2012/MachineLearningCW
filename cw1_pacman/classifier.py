# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
from sklearn.neighbors import KNeighborsClassifier


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
        
