# multiAgents.py
# --------------
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
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # avoiding ghost and go closer to food
        minGhostDist = min([manhattanDistance(ghostState.getPosition(), newPos) for ghostState in newGhostStates])
        distanceToFoods = [manhattanDistance(foodPos, newPos) for foodPos in newFood.asList()]
        minFoodDist = 0.1
        if distanceToFoods:
            minFoodDist = min(distanceToFoods)

        score = 0
        if action == 'Stop':
            score -= 50
        # avoid Stop action
        # further to ghost and closer to food is better
        return successorGameState.getScore() + minGhostDist / minFoodDist + score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        return self.value(0, gameState, 0)[1]

    def value(self, agentIndex, game_state: GameState, depth):
        if len(game_state.getLegalActions(agentIndex)) == 0 or self.depth == depth:
            return self.evaluationFunction(game_state), ""

        if agentIndex == 0:
            return self.maxValue(agentIndex, game_state, depth)
        return self.minValue(agentIndex, game_state, depth)

    def minValue(self, agentIndex, game_state: GameState, depth):
        min_value = float("inf")
        min_action = None
        for action in game_state.getLegalActions(agentIndex):
            successor_game_state = game_state.generateSuccessor(agentIndex, action)
            successor_index = agentIndex + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_depth += 1
                successor_index = 0

            cur_value = self.value(successor_index, successor_game_state, successor_depth)[0]
            if cur_value < min_value:
                min_value = cur_value
                min_action = action
        return min_value, min_action

    def maxValue(self, agentIndex, game_state: GameState, depth):
        max_value = float("-inf")
        max_action = None
        for action in game_state.getLegalActions(agentIndex):
            successor_game_state = game_state.generateSuccessor(agentIndex, action)
            successor_index = agentIndex + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_depth += 1
                successor_index = 0

            cur_value = self.value(successor_index, successor_game_state, successor_depth)[0]
            if cur_value > max_value:
                max_value = cur_value
                max_action = action
        return max_value, max_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        return self.value(0, gameState, 0, float('-inf'), float('inf'))[1]

    def value(self, agentIndex, game_state: GameState, depth, alpha, beta):
        if len(game_state.getLegalActions(agentIndex)) == 0 or self.depth == depth:
            return self.evaluationFunction(game_state), ""

        if agentIndex == 0:
            return self.maxValue(agentIndex, game_state, depth, alpha, beta)
        return self.minValue(agentIndex, game_state, depth, alpha, beta)

    def minValue(self, agentIndex, game_state: GameState, depth, alpha, beta):
        min_value = float("inf")
        min_action = None
        for action in game_state.getLegalActions(agentIndex):
            successor_game_state = game_state.generateSuccessor(agentIndex, action)
            successor_index = agentIndex + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_depth += 1
                successor_index = 0

            cur_value = self.value(successor_index, successor_game_state, successor_depth, alpha, beta)[0]
            if cur_value < alpha:
                return cur_value, action

            if cur_value < min_value:
                min_value = cur_value
                min_action = action

            beta = min(beta, cur_value)

        return min_value, min_action

    def maxValue(self, agentIndex, game_state: GameState, depth, alpha, beta):
        """
            a min will choose among the max of each node
            if we found a value that is greater than previous min
            this value is the lowest val we will choose, but the min chooser will not choose it
            so no need to calculate later nodes
        """
        max_value = float("-inf")
        max_action = None
        for action in game_state.getLegalActions(agentIndex):
            successor_game_state = game_state.generateSuccessor(agentIndex, action)
            successor_index = agentIndex + 1
            successor_depth = depth

            if successor_index == game_state.getNumAgents():
                successor_depth += 1
                successor_index = 0

            cur_value = self.value(successor_index, successor_game_state, successor_depth, alpha, beta)[0]
            if cur_value > beta:
                return cur_value, action

            if cur_value > max_value:
                max_value = cur_value
                max_action = action

            alpha = max(alpha, cur_value)

        return max_value, max_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
