# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util, sys
from game import Directions, Actions
from util import nearestPoint


#################
# Team creation #
#################


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="OffensiveReflexAgent",
    second="DefensiveReflexAgent",
):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

###########
# Globals #
###########

particleFilters = []

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        self.isOppScared = False
        self.scaredTimeLeft = 40

        self.foodLen = len(self.getFood(gameState).asList())

        if gameState.isOnRedTeam(self.index):
            self.myCapsules = gameState.getBlueCapsules()
        else:
            self.myCapsules = gameState.getRedCapsules()

        self.walls = gameState.getWalls()
        self.gridWidth = self.walls.width
        self.gridHeight = self.walls.height

        # compute mid x position
        if gameState.isOnRedTeam(self.index):
            self.midx = (self.gridWidth // 2) - 1
        else:
            self.midx = (self.gridWidth // 2)

        # compute legal positions
        self.legalPositions = []
        self.legalMidPositions = []
        for x in range(self.gridWidth):
            for y in range(self.gridHeight):
                if self.walls[x][y]:
                    continue
                pos = (x, y)
                self.legalPositions.append(pos)
                if x == self.midx:
                    self.legalMidPositions.append(pos)

        # initialize particle filters if this is offensive agent
        if isinstance(self, OffensiveReflexAgent):
            global particleFilters
            particleFilters = []
            opponents = self.getOpponents(gameState)
            for opponentIndex in opponents:
                particleFilter = ParticleFilter(opponentIndex, self.legalPositions, self.walls)
                particleFilter.initializeUniformly()
                particleFilters.append(particleFilter)

        self.doVertAStar = False
        self.vertAStarGoalPos = None

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # start = time.time()
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features["successorScore"] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def getMostLikelyOpponentPositions(self, thresh=0.1):
        dists = [particleFilter.getBeliefDistribution() for particleFilter in particleFilters]
        mostLikelyPositions = []
        for dist in dists:
            maxPosProb = 0
            mostLikelyPos = None
            for pos, prob in dist.items():
                if prob > maxPosProb:
                    maxPosProb = prob
                    mostLikelyPos = pos
            if maxPosProb >= thresh:
                mostLikelyPositions.append(mostLikelyPos)
        return mostLikelyPositions

    def getClosestDistToOpponentPacman(self, gameState):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        mostLikelyPositions = self.getMostLikelyOpponentPositions()
        closestDist = float("inf")
        for opponentPos in mostLikelyPositions:
            if gameState.isOnRedTeam(self.index):
                if opponentPos[0] > self.midx:
                    continue
            else:
                if opponentPos[0] < self.midx:
                    continue
            dist = self.getMazeDistance(myPos, opponentPos)
            closestDist = min(closestDist, dist)
        if closestDist == float("inf"):
            return None
        return closestDist

    def getClosestDistToOpponent(self, gameState):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        mostLikelyPositions = self.getMostLikelyOpponentPositions()
        closestDist = float("inf")
        for opponentPos in mostLikelyPositions:
            dist = self.getMazeDistance(myPos, opponentPos)
            closestDist = min(closestDist, dist)
        if closestDist == float("inf"):
            return None
        return closestDist

    def getPrevOpponentIndex(self):
        return (self.index + 3) % 4

    def getOpponentListIndex(self, gameState, opponentIndex):
        opponents = self.getOpponents(gameState)
        if opponentIndex == min(opponents):
            return 0
        else:
            return 1

    def displayDists(self, gameState):
        dists = []
        for i in range(4):
            if i in self.getOpponents(gameState):
                particleFilter = particleFilters[self.getOpponentListIndex(gameState, i)]
                dists.append(particleFilter.getBeliefDistribution())
            else:
                dists.append(util.Counter())
        self.displayDistributionsOverPositions(dists)

    def aStar(self, startState, isGoalState, getSuccessors, heuristic):
        node = (startState, "Null Action", 0, None)  # (state, action, path cost, parent)
        frontier = util.PriorityQueue()
        frontier.push(node, 0)
        explored = set()
        while True:
            if frontier.isEmpty():
                raise Exception, "Failure: goal not found"
            node = frontier.pop()
            if isGoalState(node[0]):
                action = node[1]
                parent_node = node[3]
                while parent_node[3] is not None: # this will throw error if startState is goalState
                    action = parent_node[1]
                    parent_node = parent_node[3]
                return action
            pos = node[0].getAgentPosition(self.index)
            if pos not in explored:
                explored.add(pos)
                for successor in getSuccessors(node[0]):
                    child_state = successor[0]
                    child_action = successor[1]
                    child_path_cost = node[2] + successor[2]
                    child_node = (child_state, child_action, child_path_cost, node)
                    frontier.push(child_node, child_path_cost + heuristic(child_state))


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # start = time.time()
        agentState = gameState.getAgentState(self.index)
        agentPos = agentState.getPosition()
        actions = gameState.getLegalActions(self.index)

        for a in self.myCapsules:
            if agentPos == a:
                self.isOppScared = True
                self.myCapsules.remove(a)
        if self.isOppScared:
            self.scaredTimeLeft -= 1
        if self.scaredTimeLeft == 0:
            self.isOppScared = False
            self.scaredTimeLeft = 40

        # particle filtering
        particleFilter = particleFilters[self.getOpponentListIndex(gameState, self.getPrevOpponentIndex())]
        particleFilter.elapseTime()
        particleFilter.observe(gameState, self.index)
        # self.displayDists(gameState)

        # use A star to go home
        if agentState.isPacman:
            isGhostNear = False
            if not self.isOppScared:
                for i in self.getOpponents(gameState):
                    opp = gameState.getAgentState(i)
                    oppPos = opp.getPosition()
                    if oppPos is not None:  # use true distance if we have it
                        dist  = self.getMazeDistance(agentPos, oppPos)
                        if dist <= 4:
                            isGhostNear = True
                            break
            foodLeft = len(self.getFood(gameState).asList())
            if not self.isOppScared:
                numCarryingThresh = 9
            else:
                numCarryingThresh = 5
            if isGhostNear or (agentState.numCarrying > numCarryingThresh) or (foodLeft <= 2):
                return self.aStar(gameState, self.goHomeIsGoalState, self.goHomeGetSuccessors, self.goHomeHeuristic)

        # use A star to move vertically if we are stuck
        if not agentState.isPacman:
            if agentPos == self.vertAStarGoalPos:
                self.doVertAStar = False
                self.vertAStarGoalPos = None
            opponentPositions = []
            for i in self.getOpponents(gameState):
                opponentPos = gameState.getAgentPosition(i)
                if opponentPos is not None:
                    opponentPositions.append(opponentPos)
            if gameState.isOnRedTeam(self.index):
                frontAgentPos = (agentPos[0] + 1, agentPos[1])
            else:
                frontAgentPos = (agentPos[0] - 1, agentPos[1])
            if self.doVertAStar or (agentPos[0] == self.midx and ((frontAgentPos in opponentPositions) or self.stayedInSamePos(gameState))):
                if not self.doVertAStar:
                    self.vertAStarStartPos = agentPos
                    validVertGoalPositions = []
                    for y in range(self.gridHeight):
                        # don't want to consider walls or current pos (this will cause A star to throw error)
                        if self.walls[self.midx][y] or agentPos == (self.midx, y):
                            continue
                        validVertGoalPositions.append((self.midx, y))
                    self.vertAStarGoalPos = random.choice(validVertGoalPositions)
                self.doVertAStar = True
                action = self.aStar(gameState, self.goVerticalIsGoalState, self.goVerticalGetSuccessors, self.goVerticalHeuristic)
                return action

        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(bestActions)

    def stayedInSamePos(self, gameState):
        prevGameState = self.getPreviousObservation()
        if prevGameState is None:
            return False
        prevPos = prevGameState.getAgentPosition(self.index)
        pos = gameState.getAgentPosition(self.index)
        if prevPos == pos:
            return True
        return False

    def closestFood(self, pos, food, walls):
        """
        closestFood -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None

    def getFeatures(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        myPos = int(myPos[0]), int(myPos[1])

        # extract the grid of food and wall locations and get the ghost locations
        food = self.getFood(gameState)
        walls = gameState.getWalls()

        features = util.Counter()

        nearGhosts = 0
        if myState.isPacman:
            for i in self.getOpponents(successor):
                opp = successor.getAgentState(i)
                oppPos = opp.getPosition()
                if oppPos is not None:  # use true distance if we have it
                    dist  = self.getMazeDistance(myPos, oppPos)
                    if dist <= 4:  # definition of near
                        nearGhosts += 1

        features["#-of-near-ghosts"] = nearGhosts

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-near-ghosts"] and food[myPos[0]][myPos[1]]:
            features["eats-food"] = 1.0

        # distance to closest food
        distToClosestFood = self.closestFood(myPos, food, walls)
        if distToClosestFood is not None:
            features["closest-food"] = distToClosestFood

        # distance to capsule
        if len(self.myCapsules) > 0:
            minDistToCapsule = float("inf")
            for capsulePos in self.myCapsules:
                distToCapsule = self.getMazeDistance(myPos, capsulePos)
                minDistToCapsule = min(minDistToCapsule, distToCapsule)
            features["distToCapsule"] = minDistToCapsule

        return features

    def getWeights(self, gameState, action):

        if self.isOppScared:
            return{
                "#-of-near-ghosts": 0,
                "eats-food": 50,
                "closest-food": -30
            }
        else:
            return {
                "#-of-near-ghosts": -150,
                "eats-food": 50,
                "closest-food": -30,
                "distToCapsule": -25
            }

    def goHomeIsGoalState(self, state):
        agentState = state.getAgentState(self.index)
        return not agentState.isPacman

    def goHomeGetSuccessors(self, state):
        curPos = state.getAgentPosition(self.index)
        successors = []
        for action in state.getLegalActions(self.index):
            nextState = self.getSuccessor(state, action)
            nextx, nexty = nextState.getAgentPosition(self.index)

            # compute min current and next distance to opponent
            minCurDist = float("inf")
            minNextDist = float("inf")
            opponents = self.getOpponents(state)
            for i in opponents:
                opponentPos = state.getAgentPosition(i)
                if opponentPos is not None:
                    curDist = self.getMazeDistance(curPos, opponentPos)
                    nextDist = self.getMazeDistance((nextx, nexty), opponentPos)
                    minCurDist = min(minCurDist, curDist)
                    minNextDist = min(minNextDist, nextDist)

            if not self.walls[nextx][nexty]:
                cost = 1
                # if action moves closer (or not at all) to opponent and opponent is near
                if minNextDist <= minCurDist and minCurDist <= 4:
                    cost = 9999
                successors.append((nextState, action, cost))
        return successors

    def goHomeHeuristic(self, state):
        pos = state.getAgentPosition(self.index)
        return min([self.getMazeDistance(pos, homePos) for homePos in self.legalMidPositions])

    def goVerticalIsGoalState(self, state):
        pos = state.getAgentPosition(self.index)
        return pos == self.vertAStarGoalPos

    def goVerticalGetSuccessors(self, state):
        successors = []
        for action in state.getLegalActions(self.index):
            nextState = self.getSuccessor(state, action)
            nextx, nexty = nextState.getAgentPosition(self.index)
            if not self.walls[nextx][nexty]:
                cost = 1
                if state.isOnRedTeam(self.index):
                    if nextx > self.midx:
                        cost = 9999
                else:
                    if nextx < self.midx:
                        cost = 9999
                successors.append((nextState, action, cost))
        return successors

    def goVerticalHeuristic(self, state):
        pos = state.getAgentPosition(self.index)
        return self.getMazeDistance(pos, self.vertAStarGoalPos)

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # start = time.time()
        actions = gameState.getLegalActions(self.index)

        # particle filtering
        particleFilter = particleFilters[self.getOpponentListIndex(gameState, self.getPrevOpponentIndex())]
        particleFilter.elapseTime()
        particleFilter.observe(gameState, self.index)
        # self.displayDists(gameState)

        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(bestActions)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)

        # Computes whether we're on defense (1) or offense (0)
        features["onDefense"] = 1
        if myState.isPacman:
            features["onDefense"] = 0

        # compute distance to opponent
        distToOpponent = self.getClosestDistToOpponent(successor)
        if distToOpponent is not None:
            features["distToOpponent"] = distToOpponent

        # computes distance to potential invaders
        mostLikelyPositions = self.getMostLikelyOpponentPositions()
        numPotentialInvaders = 0
        for pos in mostLikelyPositions:
            if gameState.isOnRedTeam(self.index):
                if pos[0] <= self.midx:
                    numPotentialInvaders += 1
            else:
                if pos[0] >= self.midx:
                    numPotentialInvaders += 1
        features["numPotentialInvaders"] = numPotentialInvaders
        if numPotentialInvaders > 0:
            potentialInvaderDist = self.getClosestDistToOpponentPacman(successor)
            if potentialInvaderDist is not None:
                features["potentialInvaderDistance"] = potentialInvaderDist

        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        if action == rev:
            features["reverse"] = 1

        # if agents kills itself
        if myState.getPosition() == self.start:
            features["dead"] = 1

        return features

    def getWeights(self, gameState, action):
        agentState = gameState.getAgentState(self.index)
        isScared = agentState.scaredTimer > 0
        if isScared:
            return{
                "numPotentialInvaders": -5,
                "onDefense": 50,
                "potentialInvaderDistance": -1,
                "reverse": -2,
                "dead": -500
            }
        else:
            return {
                "numPotentialInvaders": -100,
                "onDefense": 100,
                "potentialInvaderDistance": -10,
                "reverse": -2,
                "dead": -500,
                "distToOpponent": -4
        }


class ParticleFilter:
    def __init__(self, opponentIndex, legalPositions, walls, numParticles=1000):
        self.opponentIndex = opponentIndex
        self.legalPositions = legalPositions
        self.numParticles = numParticles
        self.walls = walls

    def initializeUniformly(self):
        self.particles = []
        for i in range(self.numParticles):
            j = i % len(self.legalPositions)
            self.particles.append(self.legalPositions[j])

    def observe(self, gameState, index):
        myState = gameState.getAgentState(index)
        myPos = myState.getPosition()
        noisyDistances = gameState.getAgentDistances()
        opponentPos = gameState.getAgentPosition(self.opponentIndex)

        # if we know the true position of the opponent
        if opponentPos is not None:
            self.particles = [opponentPos] * self.numParticles

        # observe
        weights = util.Counter()
        for p in self.particles:
            trueDistance = util.manhattanDistance(p, myPos)
            noisyDistance = noisyDistances[self.opponentIndex]
            weight = gameState.getDistanceProb(trueDistance, noisyDistance)
            weights[p] = weight

        beliefs = self.getBeliefDistribution()
        for p in beliefs:
            beliefs[p] *= weights[p]
        beliefs.normalize()

        # reinitialize particles if beliefs all have 0 prob.
        if beliefs.totalCount() == 0:
            self.initializeUniformly()
            if opponentPos is not None:
                self.particles = [opponentPos] * self.numParticles
            return

        # resample
        self.particles = []
        for _ in range(self.numParticles):
            self.particles.append(util.sample(beliefs))

    def elapseTime(self):
        newParticles = []
        for p in self.particles:
            newPosDist = self.getPositionDistributionForOpponent(p)
            newParticle = util.sample(newPosDist)
            newParticles.append(newParticle)
        self.particles = newParticles

    def getBeliefDistribution(self):
        beliefs = util.Counter()
        for p in self.particles:
            beliefs[p] += 1
        beliefs.normalize()
        return beliefs

    def getPositionDistributionForOpponent(self, opponentPos):
        # gets the uniform distribution over next positions
        nextPositions = Actions.getLegalNeighbors(opponentPos, self.walls)
        dist = util.Counter()
        for nextPos in nextPositions:
            dist[nextPos] = 1.0 / len(nextPositions)
        return dist
