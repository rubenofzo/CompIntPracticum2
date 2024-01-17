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
        closestGhostDistance = min([manhattanDistance(newPos,xy2) for xy2 in newGhostPositions])
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
        bestAction = None
        bestValue = -1000
        actions = gameState.getLegalActions(0)
        for action in actions:
            newState = gameState.generateSuccessor(0, action)
            value = minimax(self, newState, self.depth, 1)
            if value > bestValue:
               bestValue = value
               bestAction = action
        return bestAction


    # Minimax loop

def minimax(self, gameState: GameState, depth: INT, agentIndex: INT):
    numAgents = gameState.getNumAgents()
    numGhosts = numAgents - 1
    ghostValues = []
    actions = gameState.getLegalActions(agentIndex)

    #If the game has ended or a leaf node is reached the value of the current gameState is returned

    if depth == 0 or gameState.isWin() or gameState.isLose():
         return self.evaluationFunction(gameState)

     # Otherwise, the loop continues.

    else:
        # If it is pacmans turn it enters this if-statement. The value is set to a low number to help us pick the bestValue 

         if agentIndex == 0:
            bestValue = -10000
            for action in actions:
                      newNode = gameState.generateSuccessor(0, action)
                      value = minimax(self, newNode, depth, 1 % numAgents)
                      bestValue = max(bestValue, value)
            return bestValue
         else:
            bestValue = +10000 
            for action in actions:
                     newNode = gameState.generateSuccessor(agentIndex, action)
                     if agentIndex == (numAgents-1):
                         value = minimax(self, newNode, depth-1, (agentIndex + 1) % numAgents)
                     else:
                         value = minimax(self, newNode, depth, (agentIndex + 1) % numAgents)
                     bestValue = min(bestValue, value)
            ghostValues.append(bestValue)
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
        #print("max depth ",self.depth)
        #print("pacman plays")
        finalValues = []
        # first get the possible actions pacman can make from this state
        actions = gameState.getLegalActions(0)
        # the next action will happen by the ghosts
        nextAgent = 1
        alphaBorder = None
        for action in actions:
            # get the value of the miniMax tree for each action
            value = alphaBetaPrune(self,gameState.generateSuccessor(0, action),self.depth,nextAgent,alphaBorder,None)
            finalValues.append(value)
            #if alphaBorder is NULL or value > alphaBorder:
            #    alphaBorder = value
        #return the best value from all found values
        bestValue = max(finalValues)
        bestAction = actions[finalValues.index(bestValue)]
        return bestAction  

"""
start with index agent = 1 (ghosts move first)
    6 tied root test PASS
    7-1a checkdepth one ghost PASS
    2-4a vary depth PASS
    2-1a vary depth PASS
    first 4 PASS
fix indexes 
    +2-3a vary depth
    +2-2a vary depth 

let depth only increment when ghost moves
    +7-1b till 7-1c

swap when alpha and beta borders change
    +1-4

only passing beta/alpha border that you generate
    nothing

stopped expanding state if prevention happens


"""

def alphaBetaPrune(self,gameState: GameState, currDepth, agentIndex, alphaBorder, betaBorder):
    # if the state is a leaf -> return the evaluation value
    if gameState.isWin() or gameState.isLose() or currDepth <= 0:
       return self.evaluationFunction(gameState)
    # if it is not a leaf -> look at the values of the succesors
    # if its pacmans turn -> assume the worst outcome of all possible ghost moves and get that value
    if agentIndex == 0: 
       #print("pacman plays on depth ",currDepth)
       return pacmanAlphaBetaPruneLoop(self,gameState, currDepth, alphaBorder)
    #if its a ghosts turn -> assume pacman will choose the best possible action and get that value
    #print("ghosts plays on depth ",currDepth)
    return ghostAlphaBetaPruneLoop(self,gameState, currDepth, betaBorder)

#function used to handle ghosts expanding all pacman states
def ghostAlphaBetaPruneLoop(self,gameState: GameState, currDepth, previousAlfaBorder):
    # pacman will play next turn
    nextAgent = 0
    # get pacmans actions
    actions = gameState.getLegalActions(nextAgent)
    bestValue = []
    betaBorder = None
    for action in actions:
            #do action
            succesorGamestate = gameState.generateSuccessor(nextAgent, action)
            #if the alphabetaboders prevent it -> stop expening for this state
            if previousAlfaBorder is not None and betaBorder is not None and betaBorder <= previousAlfaBorder :
                return min(bestValue)
            #if the alphabeta borders dont prevent it -> expand the state
            value = alphaBetaPrune(self,succesorGamestate,currDepth-1,nextAgent,None,betaBorder) #note that we only pass the betaBorder
            # determine new alpha border
            # but only if a new value got added and if that value is bigger than the current a border 
            #             or if the alpha border was not initialised yet
            if betaBorder is None or value < betaBorder:
               betaBorder = value
            # add the value of this state
            bestValue.append(value)
    #ghosts will choose to move towards the worst pacman state 
    return min(bestValue) 

#function used to handle pacman expanding all ghost states
def pacmanAlphaBetaPruneLoop(self,gameState: GameState, currDepth, previousBetaBorder):
    # the ghosts will play next turn
    nextAgent = 1
    numberOfGhosts = gameState.getNumAgents() - 1; 
    # find all possible combinations of actions
    allActions = allCombinations(gameState,numberOfGhosts)
    bestValue = []
    alphaBorder = None
    # for each set of actions, try to expand the state 
    for actions in allActions:
            # do all actions in 'actions' for each agent
            for n in range(numberOfGhosts):
               succesorGamestate = gameState.generateSuccessor(n+1, actions[n])
            if alphaBorder is not None and previousBetaBorder is not None and alphaBorder >= previousBetaBorder:
                #if the alphabetaboders prevent it -> stop expening for this state
                #print("ignored as ",betaBorder," bigger then ",alphaBorder)
                return max(bestValue)
            # if the alphabeta borders dont prevent it -> expand the state 
            value = alphaBetaPrune(self,succesorGamestate,currDepth,nextAgent,alphaBorder,None) #note that we only pass the alphaBorder                
            # determine new beta border
            # but only if a new value got added and if that value is smaller than the current b border 
            #             or if the beta border was not initialised yet
            if alphaBorder is None or value > alphaBorder:
               alphaBorder = value
            # add the value of this state
            bestValue.append(value)
    return max(bestValue) #pacman chooses to move to the best outcome

# function used to generate all possible combinations of ghost actions
def allCombinations(gameState,numberOfGhosts):
    allActions = []
    #add all possible actions for each ghost to a list 
    # list form is [[ghost 1 actions] ... [ghost n actions]]
    for i in range(numberOfGhosts):
        newActions = gameState.getLegalActions(i+1)
        allActions.append(newActions)
    # make a list of all possible combinations of actions
    # list is of form [[ghost1Action ... ghostnAction] ... [ghost1Action ... ghostnAction]]
    allCombinations = []
    # build up the allCombinations list using a helper function and feeding it ghost actions
    for ghostActions in allActions:
        result = addOneCombination(allCombinations, ghostActions)
        allCombinations = result
    return allCombinations

# helper function for allCombinations()
def addOneCombination(initialList, toAdd): 
    # if the initial list is empty 
    # then return a list which has one list entry for each entry in xs 
    # so we can build it further when more ghost actions get added later
    if initialList == []:
        return [[x] for x in toAdd]
    # else we build a new result list which will have one entry for each x in toAdd and y in initialList
    # this way we have all possible sequences of actions for x ghosts
    result = []
    for action in toAdd:
      for newSequence in initialList:
         newSequence.append(action)
         result.append(newSequence)
    return result

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
