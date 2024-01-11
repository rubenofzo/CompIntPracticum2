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


from asyncio.windows_events import NULL
from ctypes.wintypes import BOOLEAN, INT
from xmlrpc.client import Boolean
from util import manhattanDistance
from game import Directions
import random, util
import itertools
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
        newGhostPositions = successorGameState.getGhostPositions()
        #calculate the total distance of the fastest path (using manhattan distances) between all foods
        fastestPathThroughFood = minDistanceList(newPos, newFood.asList())
        #calculate the distance to the closest ghost
        closestGhostDistance = min(map(lambda xy2 : manhattanDistance(newPos,xy2), newGhostPositions))
        #if a ghost is close and not scared => avoid succesor
        if (closestGhostDistance < 2.0 and min(newScaredTimes)==0):
            return -fastestPathThroughFood -1000
        #if not then return the negative of the distance to all foods (as a bigger number should be better)
        return -fastestPathThroughFood 

#finds the cheapest path from position to all positionsLeft
def minDistanceList(position,positionList):
    #if there are no positions to calculate distances to, the distance is 0
    if positionList == []:
        return 0
    #make a list of mahattan distances from the position to everything in the positionList
    distanceList = []
    for pos in positionList:
         distanceList.append(manhattanDistance(position,pos))
    #find the closest coordinate in the list and the distance to it
    closestDistance = min(distanceList)
    positionFound = positionList[distanceList.index(closestDistance)]
    positionList.remove(positionFound) #remove the found position from the list that is going to be searched next
    #find the cheapest path from the new found position to the left over positions
    return closestDistance + minDistanceList(positionFound,positionList)

#calculates manhattanDistance between 2 points
def manhattanDistance(xy1, xy2):
     return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


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
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        bestAction = NULL
        bestValue = -1000
        actions = gameState.getLegalPacmanActions()
        for action in actions:
            newState = gameState.generateSuccessor(0, action)
            value = minimax(self, newState, self.depth, 1)
            if value > bestValue:
               bestValue = value
               bestAction = action
        return bestAction

def minimax(self, gameState: GameState, depth: INT, agentIndex: INT):
    numAgents = gameState.getNumAgents()
    actions = gameState.getLegalActions(agentIndex)
    if depth == 0 or gameState.isWin() or gameState.isLose():
         return self.evaluationFunction
    else:
         if agentIndex == 0:
            bestValue = -10000
            for action in actions:
                      newNode = gameState.generateSuccessor(0, action)
                      value = minimax(self, newNode, depth -1, 1 % numAgents)
                      bestValue = max(bestValue, value)
            return bestValue
         else:
            bestValue = +10000
            for action in actions:
                     newNode = gameState.generateSuccessor(agentIndex, action)
                     value = minimax(self, newNode, depth -1, (agentIndex + 1) % numAgents)
                     bestValue = min(bestValue, value)
            return bestValue
         
            


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        finalValues = []
        actions = gameState.getLegalActions(0)
        betaBorder = NULL
        for action in actions:
            value = alphaBetaPrune(self,gameState.generateSuccessor(0, action),self.depth,0,0,NULL,betaBorder)
            finalValues.append(value)
            #if betaBorder is NULL or value < betaBorder:
            #    betaBorder = value
        bestValue = max(finalValues)
        bestAction = finalValues.index(bestValue)
        return bestAction  

def alphaBetaPrune(self,gameState: GameState, maxDepth, currDepth, agentIndex, alphaBorder, betaBorder):

    # if the state is a leaf -> return the evaluation value
    if gameState.isWin() or gameState.isLose() or currDepth == maxDepth:
       return self.evaluationFunction(gameState)
    # if it is not a leaf -> look at the values of the succesors
    if  agentIndex == 0: #if its pacmans turn -> just look at pacmans action and generate the tree from there
       return pacmanAlphaBetaPruneLoop(self,gameState, maxDepth, currDepth, alphaBorder, betaBorder)
    else: #if its a ghosts turn -> choose the best sequence of ghost actions (potentially a lot of actions)
       return ghostAlphaBetaPruneLoop(self,gameState, maxDepth, currDepth, alphaBorder, betaBorder)

    """
        numberOfAgents = gameState.getNumAgents();
        # if the state is a leaf -> return the evaluation value
        if gameState.isWin() or gameState.isLose() or currDepth == maxDepth:
          return self.evaluationFunction(gameState)
        # if it is not a leaf -> look at the values of the succesors
        if  agentIndex == 0: #if its pacmans turn -> just look at pacmans action and generate the tree from there
            actions = gameState.getLegalActions(agentIndex)
        else: #if its a ghosts turn -> choose the best sequence of ghost actions (potentially a lot of actions)
            actions = determineGhostActions
         
        # generate values for the succesors
        bestValue = []
        for action in actions:
            #print(agentIndex)
            succesorGamestate = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == 0  or agentIndex < numberOfAgents-1: # -1??? or +0?? 
                if alphaBorder is NULL or betaBorder is NULL or alphaBorder < betaBorder:
                   value = alphaBetaPrune(self,succesorGamestate,maxDepth,currDepth,agentIndex+1,alphaBorder,betaBorder)
            elif (alphaBorder is NULL or betaBorder is NULL) or betaBorder > alphaBorder: #  and agentIndex == numberOfAgents-1
                value = alphaBetaPrune(self,succesorGamestate,maxDepth,currDepth + 1,0,alphaBorder,betaBorder)
            # determine alpha/beta borders
            if agentIndex is not 0 and (value < betaBorder or betaBorder is NULL):
               betaBorder = value
            elif value > alphaBorder or alphaBorder is NULL:
               alphaBorder = value
            bestValue.append(value)
        # choose the MIN if you are a ghost and choose MAX if you are pacman
        if agentIndex == 0:
            return max(bestValue)
        return min(bestValue)
    """

#function used to handle ghosts expanding all pacman states
def pacmanAlphaBetaPruneLoop(self,gameState: GameState, maxDepth, currDepth, alphaBorder, betaBorder):
    actions = gameState.getLegalActions(0)
    bestValue = []
    for action in actions:
            # the value is initialised as the worst possible 
            value = 1000 
            succesorGamestate = gameState.generateSuccessor(0, action)
            #if the alphabeta borders dont prevent it -> just expand the state
            if alphaBorder is NULL or betaBorder is NULL or betaBorder > alphaBorder: 
                value = alphaBetaPrune(self,succesorGamestate,maxDepth,currDepth+1,1,alphaBorder,betaBorder)
            # determine alpha/beta borders
            if alphaBorder is NULL or value > alphaBorder:
               alphaBorder = value
            # return the value of this state
            bestValue.append(value)
    return min(bestValue)

#function used to handle pacman expanding all ghost states
def ghostAlphaBetaPruneLoop(self,gameState: GameState, maxDepth, currDepth, alphaBorder, betaBorder):
    numberOfAgents = gameState.getNumAgents(); 
    # find all possible combinations of actions
    allActions = allCombinations(gameState,numberOfAgents)
    bestValue = []
    # for each set of actions, try to expand the state 
    for actions in allActions:
            # the value is initialised as the worst possible 
            value = -1000
            # do all actions
            for n in range(numberOfAgents-1):
               succesorGamestate = gameState.generateSuccessor(n+1, actions[n])
            # expand a state if one of the borders is uninnitialised and if the alphabetaborders don't prevent it
            if alphaBorder is NULL or betaBorder is NULL or alphaBorder < betaBorder:
                 value = alphaBetaPrune(self,succesorGamestate,maxDepth,currDepth+1,0,alphaBorder,betaBorder)
            # determine new beta border
            if betaBorder is NULL or value < betaBorder:
               betaBorder = value
            bestValue.append(value)
    return max(bestValue)

def allCombinations(gameState,numberOfAgents):
    xss = []
    for i in range(numberOfAgents-1):
        newActions = gameState.getLegalActions(i+1)
        xss.append(newActions)
    #print("all actions", xss)
    ys = []
    for xs in xss:
        result = addOneCombination(ys, xs)
        ys = result
    return ys


def addOneCombination(ys, xs): 
    if ys == []:
      for x in xs:
        ys.append([x])
      return ys
    zs = []
    for x in xs:
      for y in ys:
         #print("y is ",y)
         #print("x is ",x)
         y.append(x)
         zs.append(y)
         #print("together is ",zs)
    return zs


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
