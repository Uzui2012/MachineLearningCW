# MSc Machine Learning Group Coursework

## Coursework 1: "Sophisticated" Pacman Game Classifier
Built a simple shallow artificial neural network with a single hidden layer that trains on a data set of observations made by pacman that are arbitrary to us with the respective "correct" move that should be followed. 

The following command from within the cw1_pacman folder is to be made to use the ClassifierAgent.

<code>python pacman.py --pacman ClassifierAgent</code>


## Coursework 2: Q-Learning Agent

Use the following command to perform a <b>*single*</b> training episode, thanks to the limitations of using the _final()_ funciton to initialise the q-table.  

<code>python pacman.py -p QLearnAgent -l smallGrid -x 2 -n 2</code>