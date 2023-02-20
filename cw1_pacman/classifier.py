# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time

class Classifier:
    # Classifier is initialised with nothing passed through. Only initialise 
    # object scope constants and variables here. We may not even use this.
    def __init__(self):
        self.LEARNING_RATE = 0.01
        self.THETA = 0.8
        pass

    # As far as I can see, we don't have to perform any retraining on the model
    # between sets of games, unless we know the environment changes (I believe
    # it doesn't). Again we may not even use this.
    def reset(self):
        pass
    
    # Trains on training dataset, the arrays of input <data>, on target, or 
    # output singular integer <target>.
    #
    # <data> = [[001011101010...], [001011101010...], ...]
    # <target> = array of (0 or 1 or 2 or 3)'s
    #
    # This happens before classifier is run on the environment.
    def fit(self, data, target):
        inputSize = len(data[0])
        self.model = MLP(inputSize, inputSize + 5, 4)
        for i in range(len(data)):
            prediction = self.model.forward(data[i], train = True)

        self.model.backward(theta = self.THETA, 
                            learningRate = self.LEARNING_RATE,  
                            targets = target)
        pass

    # We simply output the integer output/prediction given a singular feature
    # array. Current legal moves are given as array of string 'North', 'East',
    # 'South', 'West', and 'Stop'.
    #
    # ClassifierAgent.py handles output conversion of the number to these 
    # strings taken by the environment. Aswell as performing a random action if
    # our selected move is illegal anyway. So we may not even use legal at all.
    
    def predict(self, data, legal=None):
        prediction = self.model.forward(data, train = False)
        print(prediction)
        # Currently only outputs 1 (pretty sure that means East). 
        return prediction
        
# Multilayer Perception Class
# Will have singular of both forward and backward passes.
# Will perform batch back progagation. 
class MLP:
    def __init__(self, inputSize, hiddenSize, outputSize = 4):
        self.inputSize = inputSize # unaccounted bias
        self.hiddenSize = hiddenSize # unaccounted bias
        self.outputSize = outputSize
        # Input i, output node j.
        # Note: input and hidden sizes are FLIPPED for easier notation during
        # calculation later on.
        self.weightsJI = np.random.uniform(low = -1, high = 1,
                                       size = (self.hiddenSize, 
                                               self.inputSize + 1))
        
        # Input node j, output k
        # Note: hidden and output sizes are FLIPPED for easier notation during
        # calculation later on.
        self.weightsKJ = np.random.uniform(low = -1, high = 1,
                                       size = (self.outputSize, 
                                               self.hiddenSize + 1))
    
        self.batch = []

    # ReLu activation function
    def activationReLu(self, x):
        return max(0.0, x)

    # Derivative of ReLu 
    # (Note, we return 0 when x is 0 despite d/dx being undefined at x = 0)
    def derivativeReLu(self, x):
        if x > 0:
            return 1
        else:
            return 0

    # Sigmoid activation function
    def activationSig(self, x):
        return 1/(1 + np.exp(-x))

    # Derivative of sigmoid activation funciton
    def derivativeSig(self, x):
        return (x * (1.0 - x))

    # Initial part of forward pass, iterates over all inputs for the output
    # hidden node j. Includes activation function use.
    def sumInputOnWeights(self, input, j):
        temp = self.weightsJI[j][0] # w_j_0
        for i, x in enumerate(input):
            temp += x * self.weightsJI[j][i + 1]
        return self.activationReLu(temp)

    # 2nd part of forward pass, iterates over all y_j for the output k node z_k.
    # Includes activation functions use.
    def sumHiddenOnWeights(self, y, k):
        temp = self.weightsKJ[k][0] # w_k_0
        for j, y_j in enumerate(y):
            temp += y_j * self.weightsKJ[k][j + 1]
        return self.activationSig(temp)

    def forward(self, input, train = False):
        # z_k = act_func( w_k_0 + 
        #       sum_over_hidden_nodes( 
        #           weightsKJ * act_func( w_j_0 + 
        #           sum_over_inputs(weightsJI * x_i ))))
        
        y = np.zeros(self.hiddenSize)
        # Calculate all outputs of hidden nodes, y
        for j, y_j in enumerate(y):
            y[j] = self.sumInputOnWeights(input, j)

        z = np.zeros(self.outputSize)
        # Calculate all outputs of output nodes, z
        for k, z_k in enumerate(z):
            z_k = self.sumHiddenOnWeights(y, k)
        # Return index of the maximum probability selection.
        # Can change to perform a random selection, or any other acceptable 
        # method.
        if train:
            self.batch.append((y, z))
        return np.argmax(z)
        
    def crossEntropyLossBatch(predictions, targets, epsilon):
        preds = np.clip(predictions, epsilon, 1. - eplsion)
        N = preds.shape[0]
        loss = -np.sum(targets * np.log(preds + 0.000000001)) / N
        return loss

    def meanSqLoss(self, pred, target):
        return np.sum(np.square(pred-target)) / (2 * self.inputSize)

    def backward(self, theta, learningRate, targets):
        for batchIdx, batchParts in enumerate(self.batch):
            error = np.argmax(batchParts[1]) - targets[batchIdx]
            dk = error * self.derivativeSig(np.argmax(batchParts[1]))
            temp = 0
            print(self.weightsKJ)
            time.sleep(10)
            for i in range(4):
                temp += dk*self.weightsKJ[i]
            dj = 0
            for y_i in batchParts[0]:
                dj += temp * self.derivativeReLu(y_i)

            #print(dk)
            print(dj)
            






    
    