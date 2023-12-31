# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    stack = util.Stack()
    visited = set()
    curNode = [problem.getStartState(), []]
    while not problem.isGoalState(curNode[0]):
        curState, curPlan = curNode
        visited.add(curState)
        for state, action, cost in reversed(problem.getSuccessors(curState)):
            if state in visited:
                continue
            newPlan = curPlan.copy()
            newPlan.append(action)
            stack.push([state, newPlan])

        if stack.isEmpty():
            break
        curNode = stack.pop()

    return curNode[1]


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    queue, visited, curNode = util.Queue(), set(), [problem.getStartState(), []]
    while not problem.isGoalState(curNode[0]):
        curState, curPlan = curNode
        visited.add(curState)
        for state, action, cost in problem.getSuccessors(curState):
            if state in visited:
                continue
            newPlan = curPlan.copy()
            newPlan.append(action)
            queue.push([state, newPlan])

        if queue.isEmpty():
            break
        while curNode[0] in visited and not queue.isEmpty():
            curNode = queue.pop()

    return curNode[1]


def uniformCostSearch(problem: SearchProblem) -> list:
    """Search the node of the least total cost first."""
    priorityQueue = util.PriorityQueue()
    visited = set()
    curNode = [problem.getStartState(), [], 0]  # [state, plan, cost]
    while not problem.isGoalState(curNode[0]):
        curState, curPlan, curCost = curNode
        visited.add(curState)
        for state, action, cost in problem.getSuccessors(curState):
            if state in visited:
                continue
            newPlan = curPlan.copy()
            newPlan.append(action)
            newCost = curCost + cost
            priorityQueue.push([state, newPlan, curCost + cost], newCost)

        if priorityQueue.isEmpty():
            break

        """
        # update curNode
        # since i dont use a set（） to track if a state is in the priority queue and if the one in the priority queue
        # has cheaper cost
        # this loop does that implicitly, bc the cheapest duplicated state in the priority queue got popped first
        # and added to visited set(), if found the curNode already visited, meaning there was a better path to this node
        # and we should disregard it
        """
        while curNode[0] in visited and not priorityQueue.isEmpty():
            curNode = priorityQueue.pop()

    return curNode[1]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # not using util.PriorityQueueWithFunction() bc cannot store [state, plan, cost, cost + heuristic] as item
    priorityQueue = util.PriorityQueue()
    visited = set()

    # [state, plan, cost, cost + heuristic]
    curNode = [problem.getStartState(), [], 0, heuristic(problem.getStartState(), problem)]
    while not problem.isGoalState(curNode[0]):
        curState, curPlan, curCost, estimate = curNode
        visited.add(curState)
        for state, action, cost in problem.getSuccessors(curState):
            if state in visited:
                continue
            newPlan = curPlan.copy()
            newPlan.append(action)
            newCost = curCost + cost
            newEstimate = newCost + heuristic(state, problem)
            priorityQueue.push([state, newPlan, newCost, newEstimate], newEstimate)

        if priorityQueue.isEmpty():
            break

        while curNode[0] in visited and not priorityQueue.isEmpty():
            curNode = priorityQueue.pop()

    return curNode[1]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
