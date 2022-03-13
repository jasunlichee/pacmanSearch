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


from cmath import inf
from json.encoder import INFINITY
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        newPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood().asList()

        minFoodist = float('inf')
        for food in newFood:
            minFoodist = min(minFoodist, manhattanDistance(newPos, food))

        ghostDist = 0
        for ghost in currentGameState.getGhostPositions():
            ghostDist = manhattanDistance(newPos, ghost)
            if (ghostDist < 2):
                return -float('inf')

        foodLeft = currentGameState.getNumFood()
        capsLeft = len(currentGameState.getCapsules())

        foodLeftMultiplier = 950050
        capsLeftMultiplier = 10000
        foodDistMultiplier = 950

        additionalFactors = 0
        if currentGameState.isLose():
            additionalFactors -= 50000
        elif currentGameState.isWin():
            additionalFactors += 50000

        return 1.0/(foodLeft + 1) * foodLeftMultiplier + ghostDist + \
            1.0/(minFoodist + 1) * foodDistMultiplier + \
            1.0/(capsLeft + 1) * capsLeftMultiplier + additionalFactors

def scoreEvaluationFunction(currentGameState, index):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()[index]

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, index = 0, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = index # Pacman is always agent index 0
        self.evaluationFunction = lambda state:util.lookup(evalFn, globals())(state, self.index)
        self.depth = int(depth)

class MultiPacmanAgent(MultiAgentSearchAgent):
    def minimax(self, currentGameState, depth, agent):
        if currentGameState.isWin() or currentGameState.isLose() or depth == 0:
            return [self.eval(currentGameState), None]
        if agent == 0:
            maxEval = -float(inf)
            maxAction = Directions.STOP
            for action in currentGameState.getLegalActions(agent):
                eval = self.minimax(currentGameState.generateSuccessor(agent, action), depth - 1, 1)[0]
                if(eval > maxEval):
                    maxAction = action
                    maxEval = eval
            return [maxEval, maxAction]
        else:
            minEval = float(inf)
            next = agent + 1
            if currentGameState.getNumGhosts() == next:
                next = 0
            if next == 0:
                depth = depth - 1
            for action in currentGameState.getLegalActions(agent):
                eval = self.minimax(currentGameState.generateSuccessor(agent, action), depth, next)[0]
                if(eval < minEval):
                    minAction = action
                    minEval = eval
            return [minEval, minAction]
    
    def getAction(self, gameState):
        index = self.index # pacman index
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.

        Some functions you may need:
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalMoves = gameState.getLegalActions(agent)
        legalNextState = [gameState.generateSuccessor(agent, action)
                          for action in legalMoves]
        """
        "*** YOUR CODE HERE ***"
        #print("Number of Pacmans:", gameState.getNumPacman(), ", Number of ghosts:", gameState.getNumGhosts())
        return self.minimax(gameState, self.depth, 0)[1]

    def eval(self, currentGameState):
        fooddist = float(inf)
        for food in currentGameState.getFood().asList():
            fooddist = min(fooddist, manhattanDistance(food, currentGameState.getPacmanPosition(0)))

        ghostDist = 0
        for ghost in currentGameState.getGhostPositions():
            ghostDist = manhattanDistance(ghost, currentGameState.getPacmanPosition(0))
            if (ghostDist < 3):
                return -float(inf)

        return 1000000.0/(currentGameState.getNumFood() + 1) + ghostDist + 1000.0/(fooddist + 1) + 10000.0/(len(currentGameState.getCapsules()) + 1)
        
class RandomAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions(self.index)
        return random.choice(legalMoves)




