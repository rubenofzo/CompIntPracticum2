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


from argparse import Action
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
        if (action == "stop"):
            return -100

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = successorGameState.getGhostPositions()
        #calculate the distance to the closest ghost
        closestGhostDistance = min([manhattanDistance(newPos,xy2) for xy2 in newGhostPositions])
        #if a ghost is close and not scared => avoid succesor
        if (closestGhostDistance <= 2.0 and min(newScaredTimes)==0):
            return -1000
        newFood = successorGameState.getFood()
        #calculate the total distance of the fastest path (using manhattan distances) between all foods
        fastestPathThroughFood = minDistanceList(newPos, newFood.asList())
        #if not then return the negative of the distance to all foods (as a bigger number should be better)
        return -fastestPathThroughFood  

#finds the cheapest path from position to all positionsLeft
def minDistanceList(position,positionList):
    #if there are no positions to calculate distances to, the distance is 0
    if positionList == []:
        return 0
    #make a list of mahattan distances from the position to everything in the positionList
    distanceList = [manhattanDistance(position,pos) for pos in positionList]
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

        #Some initialisation values

        bestAction = None
        bestValue = -1000
        actions = gameState.getLegalActions(0)

        #For all possible moves of pac man

        for action in actions:
            newState = gameState.generateSuccessor(0, action)
            value = minimax(self, newState, self.depth, 1)

            # Checks if the current value is the bestValue and sets current action to bestAction

            if value > bestValue:
               bestValue = value
               bestAction = action

        return bestAction


    # Minimax loop

def minimax(self, gameState: GameState, depth: INT, agentIndex: INT):
    numAgents = gameState.getNumAgents()
    
    actions = gameState.getLegalActions(agentIndex)

    #If the game has ended or a leaf node is reached the value of the current gameState is returned

    if depth == 0 or gameState.isWin() or gameState.isLose():
         return self.evaluationFunction(gameState)

     # Otherwise, the loop continues.

    else:
        # If it is pacmans turn it enters this if-statement. The value is set to a low number to help us pick the bestValue with the 'max' function

         if agentIndex == 0:
            bestValue = -10000

            # For all possible actions of pacman

            for action in actions:

                      #Expand the node

                      newNode = gameState.generateSuccessor(0, action)

                      #Calculate new value by applying expectimax to the successor

                      value = minimax(self, newNode, depth, 1 % numAgents)

                      # Current bestValue, compares with the new value

                      bestValue = max(bestValue, value)
            return bestValue

        #Case of a ghost:

         else:
            bestValue = +10000 

            # For all possible actions of the ghost

            for action in actions:
                     newNode = gameState.generateSuccessor(agentIndex, action)

                      #If it is the last ghost: Edit depth and calculate value of successor

                     if agentIndex == (numAgents-1):
                         value = minimax(self, newNode, depth-1, (agentIndex + 1) % numAgents)

                      # Otherwise, keep this depth and calculate value of successor

                     else:
                         value = minimax(self, newNode, depth, (agentIndex + 1) % numAgents)
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
        bestAction = None
        bestValue = -1000
        # first get the possible actions pacman can make from this state
        actions = gameState.getLegalActions(0)
        # the next action will happen by the ghosts
        nextAgent = 1
        alphaBorder = None
        for action in actions:
            # get the value of the miniMax tree for each action
            newState = gameState.generateSuccessor(0, action)
            value = alphaBetaPrune(self,newState,self.depth,nextAgent,alphaBorder,None)
            if value > bestValue:
               bestValue = value
               bestAction = action
            # update alphaBorder
            if alphaBorder is None or value > alphaBorder:
                alphaBorder = value
        #return the best value from all found values
        return bestAction       

def alphaBetaPrune(self,gameState: GameState, currDepth, agentIndex, alphaBorder, betaBorder):
    # if the state is a leaf -> return the evaluation value
    if gameState.isWin() or gameState.isLose() or currDepth <= 0:
       return self.evaluationFunction(gameState)
    # if it is not a leaf -> look at the values of the succesors
    # if its pacmans turn -> assume pacman will choose the best possible action and get that value 
    if agentIndex == 0: 
       #print("pacman plays on depth ",currDepth)
       return pacmanAlphaBetaPruneLoop(self,gameState, currDepth, alphaBorder, betaBorder)
    #if its a ghosts turn -> assume the worst outcome of all possible ghost moves and get that value
    #print("ghosts plays on depth ",currDepth)
    return ghostAlphaBetaPruneLoop(self,gameState, currDepth, alphaBorder,betaBorder,agentIndex)

#function used to handle ghost state expansion 
def ghostAlphaBetaPruneLoop(self,gameState: GameState, currDepth, previousAlfaBorder,previousBetaBorder,agentIndex):
    numAgents = gameState.getNumAgents()
    numGhosts = numAgents - 1
    actions = gameState.getLegalActions(agentIndex)
    bestValue = 10000 
    betaBorder = previousBetaBorder
    for action in actions:
        #if the alphabetaboders prevent it -> stop expening for this state
        if previousAlfaBorder is not None and betaBorder is not None and betaBorder < previousAlfaBorder:
            #print("ignored as ",betaBorder," bigger then ",alphaBorder)
            return bestValue
        #expande node
        newNode = gameState.generateSuccessor(agentIndex, action)
        #recurse
        if agentIndex == numGhosts:
            value = alphaBetaPrune(self, newNode, currDepth-1, (agentIndex + 1) % numAgents,previousAlfaBorder,betaBorder)
        else:
            value = alphaBetaPrune(self, newNode, currDepth, (agentIndex + 1) % numAgents,previousAlfaBorder,betaBorder)
        bestValue = min(bestValue, value)
        
        # determine new beta border
        if betaBorder is None or value < betaBorder:
            betaBorder = value
    return bestValue

#function used to handle pacman state expansion
def pacmanAlphaBetaPruneLoop(self,gameState: GameState, currDepth, previousAlfaBorder, previousBetaBorder):
    numAgents = gameState.getNumAgents()
    actions = gameState.getLegalActions(0)
    bestValue = -10000
    alphaBorder = previousAlfaBorder
    for action in actions:
        #if the alphabetaboders prevent it -> stop expening for this state
        if alphaBorder is not None and previousBetaBorder is not None and alphaBorder > previousBetaBorder:
            return bestValue
        #Pacman plays
        newNode = gameState.generateSuccessor(0, action)
        value = alphaBetaPrune(self, newNode, currDepth, 1 % numAgents,alphaBorder,previousBetaBorder)
        bestValue = max(bestValue, value)

        # determine new alfa border
        if alphaBorder is None or value > alphaBorder:
                alphaBorder = bestValue
    return bestValue

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
        bestAction = None
        bestValue = -1000
        actions = gameState.getLegalActions(0)
        for action in actions:
            newState = gameState.generateSuccessor(0, action)
            value = expectimax(self, newState, self.depth, 1)

            # Makes sure all directions other than stop gets preferred and updates the value

            if value >= bestValue and action != Directions.STOP:
                bestValue = value
                bestAction = action
        return bestAction

def expectimax(self, gameState: GameState, depth: INT, agentIndex: INT):
    # Gets the number of agents (necessary for calculating the number of ghosts)
    numAgents = gameState.getNumAgents()

    # Creates list for keeping all ghost values

    ghostValues = []

    #Gets all the legal actions of the current agent

    actions = gameState.getLegalActions(agentIndex)

    #If the game has ended or a leaf node is reached the value of the current gameState is returned

    if depth == 0 or gameState.isWin() or gameState.isLose():
         return self.evaluationFunction(gameState)

     # Otherwise, the loop continues.

    else:
        # If it is pacmans turn it enters this if-statement. The value is set to a low number to help us pick the bestValue with the 'max' function

         if agentIndex == 0:
            bestValue = -10000

            # For all possible actions of pacman

            for action in actions:

                      # Expand the node

                      newNode = gameState.generateSuccessor(0, action)

                      #Calculate new value by applying expectimax to the successor

                      value = (expectimax(self, newNode, depth, 1 % numAgents) - 5)

                      # Current bestValue, compares with the new value

                      bestValue = max(bestValue, value)
            return bestValue

        # Case of the agentIndex being a ghost:

         else:            

            #For each action possible of the current ghost

            for action in actions:
                     newNode = gameState.generateSuccessor(agentIndex, action)

                     #If it is the last ghost: Edit depth and calculate value of successor

                     if agentIndex == (numAgents-1):
                         value = expectimax(self, newNode, depth-1, (agentIndex + 1) % numAgents)
                         ghostValues.append(value)
                     
                     # Otherwise, keep this depth and calculate value of successor

                     else:
                         value = minimax(self, newNode, depth, (agentIndex + 1) % numAgents)

                         #Add the current value to the list keeping all ghostvalues

                         ghostValues.append(value)
            
            # Returns the average of all the ghostvalues 
            return (sum(ghostValues)/len(ghostValues))

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    def evaluationFunction(currentGameState: GameState, action):
        if (action == "stop"):
            return -100

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = successorGameState.getGhostPositions()
        #calculate the distance to the closest ghost
        closestGhostDistance = min([manhattanDistance(newPos,xy2) for xy2 in newGhostPositions])
        #if a ghost is close and not scared => avoid succesor
        if (closestGhostDistance <= 2 and min(newScaredTimes)== 0):
            return -1000
        newFood = successorGameState.getFood()
        #calculate the total distance of the fastest path (using manhattan distances) between all foods
        fastestPathThroughFood = minDistanceList(newPos, newFood.asList())
        #if not then return the negative of the distance to all foods (as a bigger number should be better)
        return -fastestPathThroughFood 

    # get all possible pacman actions
    legalMoves = currentGameState.getLegalActions()

    # if actions are given => evaluate the actions and return the best one
    if legalMoves != []:
        scores = [evaluationFunction(currentGameState, action) for action in legalMoves]    
        return max(scores)
    # otherwise look if you are dead or alive
    newPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newGhostPositions = currentGameState.getGhostPositions()
    # determine closest ghost
    closestGhostDistance = min([manhattanDistance(newPos,ghostPos) for ghostPos in newGhostPositions])
    # if a ghost is very close we are dead => avoid this state
    if min(newScaredTimes)==0 and closestGhostDistance == 0:
        return -1000
    # if a ghost is not close we win => try to go to this state
    return 1000

    

# Abbreviation
better = betterEvaluationFunction
