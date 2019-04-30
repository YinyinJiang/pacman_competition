# pacmanAgents.py
# ---------------
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
from game import Agent
import random
import math


class CompetitionAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        legal = state.getLegalPacmanActions()

        # root
        v0 = Node(0, 0, [], [], None, legal)

        while True:
            v1 = self.treePolicy(v0)

            reward, flag2 = self.defaultPolicy(state, v1)
            if not flag2:
                break

            self.backUp(v1, reward)
        return self.mostVisitedChild(v0).action_seq[0]

    # selection
    def treePolicy(self, v0):
        while True:
            if len(v0.unvisited) > 0:
                return self.expansion(v0)
            else:
                tmp = self.bestChild(v0, 1.5)
                if tmp == v0:
                    break
                else:
                    v0 = tmp
        return v0

    def expansion(self, v0):
        v1 = Node(0, 0, [], [], None, [])
        un_act = v0.unvisited.pop()
        v1.action_seq.append(un_act)
        v1.parent = v0
        v0.children.append(v1)
        return v1

    def bestChild(self, v, c):
        arg_max = 0
        best_child = v
        for i in range(0, len(v.children)):
            v1 = v.children[i]
            Q_v1 = v1.Q_reward
            N_v1 = v1.N_numOfVisit
            N_v = v.N_numOfVisit
            tmp_max = float(Q_v1) / float(N_v1) + float(c) * math.sqrt(2. * math.log(float(N_v)) / float(N_v1))
            if tmp_max > arg_max:
                arg_max = tmp_max
                best_child = v1
        return best_child

    # simulation
    def defaultPolicy(self, state, v1):
        tempActionSeq = v1.action_seq
        next_state = self.getState(state, tempActionSeq)
        for i in range(0, 5):
            if next_state is None:
                return None, False
            elif next_state.isWin():
                return 10000., True
            elif next_state.isLose():
                return -1000., True
            else:
                possible = next_state.getAllPossibleActions()
                ran_action = possible[random.randint(0, len(possible) - 1)]
                next_state = next_state.generatePacmanSuccessor(ran_action)
        if next_state is None:
            return None, False
        elif next_state.isWin():
            return 10000., True
        elif next_state.isLose():
            return -1000., True
        return self.gameEvaluation(state, next_state), True

    def getState(self, state, actions):
        next_state = state
        for i in range(0, len(actions)):
            if next_state is None:
                return None
            elif next_state.isLose() or next_state.isWin():
                return next_state
            else:
                next_state = next_state.generatePacmanSuccessor(actions[i])
        return next_state

    def backUp(self, v, reward):
        while v is not None:
            v.N_numOfVisit = v.N_numOfVisit + 1
            v.Q_reward = v.Q_reward + reward
            v = v.parent

    def mostVisitedChild(self, v0):
        largest_N = 0
        res = []
        for i in range(0, len(v0.children)):
            if v0.children[i].N_numOfVisit > largest_N:
                largest_N = v0.children[i].N_numOfVisit
                res = []
                res.append(v0.children[i])
            elif v0.children[i].N_numOfVisit == largest_N:
                res.append(v0.children[i])
        return res[random.randint(0, len(res) - 1)]

    def gameEvaluation(self, state1, state2):
        foodScore = (state1.getNumFood() - state2.getNumFood())
        capsuleScore = (len(state1.getCapsules()) - len(state2.getCapsules())) * 500.
        ghostScore = 1000. if state2.getScore() - state1.getScore() - foodScore >= 200 else 0.
        return foodScore + capsuleScore + ghostScore

class Node(object):
    def __init__(self, Q_reward, N_numOfVisit, action_seq, children, parent, unvisited):
        self.Q_reward = Q_reward
        self.N_numOfVisit = N_numOfVisit
        self.action_seq = action_seq
        self.children = children
        self.parent = parent
        self.unvisited = unvisited