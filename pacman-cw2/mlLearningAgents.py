# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

# Wrapper class that encapsulates the game state given by a GameState object.
#
class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        self.legal_actions = state.getLegalActions() # Easy legal action storage
        self.state = self.concatState(state) 
        
    # Full deep copy of the state, however the score is always 0. We found 
    # that if we are storing the entire GameState as a part of our q-table dict- 
    # type keys then the unique scores do not allow overlap and just produce a
    # unique entry to the q-table. Restoring any score to 0 doesn't prevent our
    # score aquisition in the QLearnAgent.computeReward() method as we do this 
    # to the original GameState class and not our wrapper.
    def concatState(self, state: GameState):
        temp = state.deepCopy()
        temp.data.score = 0
        return temp
        
class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        self.q_value = util.Counter()
        self.counts = util.Counter()

        # Store previous states
        self.lastState = None
        self.lastAction = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # Computes the reward produced when moving from one state to another.
    # Simply computing the difference in score of the two states gives a simple,
    # but sophisticated, reward function.
    # 
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        return endState.getScore() - startState.getScore()

    # Simply returns the Q value in the Q Table given the state-action pair.
    # 
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        return self.q_value[(state.state, action)]

    # Returns the maximum Q value of a state across all the actions available to
    # it.
    #
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        q_list = []
        for a in state.legal_actions:
            q = self.getQValue(state, a)
            q_list.append(q)

        if len(q_list) == 0:
            return 0

        return max(q_list)

    # Performs a Q-Learning update to the internal model.
    #
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        current_qvalue = self.getQValue(state, action)
        max_qvalue = self.maxQValue(nextState)
        estimate = reward + self.gamma * max_qvalue - current_qvalue
        self.q_value[(state.state, action)] = current_qvalue + self.alpha * estimate

    # Increments the stored visitation counts.
    #
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        self.counts[(state.state, action)] += 1

    # Count getter method, returning the number of times a given action has been
    # taken in a given state.
    #
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        return self.counts[(state.state, action)]

    # Exploration function, computing a value based on the number of times an 
    # action has been taken.
    #
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"

        if counts < self.maxAttempts:
            return 1000.0
        else:
            return utility

    # Chooses the action to take using the Q Learning algorithm.
    #
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state)

        # Update the Q-value for the previous state-action pair and the current state
        if self.lastState is not None and self.lastAction is not None:
            s = GameStateFeatures(self.lastState)
            reward = self.computeReward(self.lastState, state)
            self.learn(s, self.lastAction, reward, stateFeatures)
            self.updateCount(s, self.lastAction)

        utility = [self.explorationFn(self.getQValue(stateFeatures, action), self.getCount(stateFeatures, action)) for action in legal]
        action = legal[utility.index(max(utility))]

        self.lastState = state
        self.lastAction = action

        return action

    # Method that is called upon a game terminating.
    #
    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)

        stateFeatures = GameStateFeatures(state)
        s = GameStateFeatures(self.lastState)
        reward = self.computeReward(self.lastState, state)
        self.learn(s, self.lastAction, reward, stateFeatures)
        self.updateCount(s, self.lastAction)
