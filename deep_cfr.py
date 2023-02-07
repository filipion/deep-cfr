import random

from tensorflow import keras
import numpy as np
import pandas as pd

PASS = 0
BET = 1
NUM_ACTIONS = 2
nodeMap = {}

regretNet = keras.Sequential([
    keras.layers.Dense(12, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])

strategyNet = keras.Sequential([
    keras.layers.Dense(12, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])

def getStrategy(infoSet, realizationWeight):
    encoder = KuhnEncoder()
    return regretNet(encoder.encode(infoSet))

def getAverageStrategy(infoSet, realizationWeight):
    encoder = KuhnEncoder()
    return strategyNet(encoder.encode(infoSet))

class KuhnEncoder():
    def __init__(self):
        self.infostates = pd.Series(["1", "2", "3", "1p", "2p", "3p", "1b", "2b", "3b", "1pb", "2pb", "3pb"])
    
    def encode(self, infoState):
        print(self.infostates, infoState)
        return np.array(self.infostates == infoState).astype(float)
    

class KuhnNode():
    def __init__(self, infoSet):
        self.infoSet = infoSet

    def __repr__(self):
        return "CFR node for the infoset {infoset}. Strategy: {strategy}".format(infoset=self.infoSet, 
            strategy=["{:.2f}".format(x) for x in self.getAverageStrategy()])

    def IsTerminal(cards, history):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player
        if plays > 1 :
            terminalPass = history[-1] == 'p'
            doubleBet = history[-2:] == "bb"
            isPlayerCardHigher = cards[player] > cards[opponent]
            if terminalPass:
                if history == "pp":
                    return 1 if isPlayerCardHigher else -1
                else:
                    return 1
            elif doubleBet:
                return 2 if isPlayerCardHigher else -2
        return None

    def saveToBuffer():
        # TODO
        pass

def cfr(cards, history, p0, p1, time):
    plays = len(history)
    player = plays % 2
    if player == 0:
        realizationWeight = p0
    else:
        realizationWeight = p1

    if(KuhnNode.IsTerminal(cards, history) is not None):
        return KuhnNode.IsTerminal(cards, history)
    
    infoSet = str(cards[player]) + history
    if not infoSet in nodeMap:
        nodeMap[infoSet] = KuhnNode(infoSet)
    node = nodeMap[infoSet]
    strategy = node.getStrategy(realizationWeight)

    nodeUtil = 0
    util = [0 for _ in range(NUM_ACTIONS)]
    for a in range(NUM_ACTIONS):
        nextHistory = history + ('p' if a == PASS else 'b')
        if player == 0:
            util[a] = -cfr(cards, nextHistory, p0 * strategy[a], p1, time)
        else:
            util[a] = -cfr(cards, nextHistory, p0, p1 * strategy[a], time)
        nodeUtil += util[a] * strategy[a]

    for a in range(NUM_ACTIONS):
        regret = util[a] - nodeUtil
        node.regretSum[a] += ((p1 if player == 0 else p0) * regret) # pi_i_negative
    
    return nodeUtil

# TODO 
# Redo this
def train(iterations):
    util = 0
    for i in range(iterations):
        cards = [1, 2, 3]
        random.shuffle(cards)
        util += cfr(cards, '', 1.0, 1.0, i)
        if(i % (iterations // 10) == 0):
            print("Iteration number: {}/{}".format(i, iterations))
            print("Avg, utility for player 1: {:.4f}".format(util / (i + 1)))

def logInfosets():
    infoSets = sorted(nodeMap.keys())
    for node in infoSets:
        print(nodeMap[node])
  