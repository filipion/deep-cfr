import random

from tensorflow import keras
import numpy as np
import pandas as pd

from  buffer import ReplayBuffer

PASS = 0
BET = 1
NUM_ACTIONS = 2
NUM_PLAYERS = 2
nodeMap = {}

regretNet = keras.Sequential([
    keras.layers.Normalization(input_shape=[12, ]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])

strategyNet = keras.Sequential([
    keras.layers.Normalization(input_shape=[12, ]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])

def getStrategy(infoSet):
    encoder = KuhnEncoder()
    return regretNet(encoder.encode(infoSet))

def getAverageStrategy(infoSet):
    encoder = KuhnEncoder()
    return strategyNet(encoder.encode(infoSet))

def getSample(strategy):
    return


class KuhnEncoder():
    def __init__(self):
        self.infostates = pd.Series(["1", "2", "3", "1p", "2p", "3p", "1b", "2b", "3b", "1pb", "2pb", "3pb"])
    
    def encode(self, infoState):
        return np.array(self.infostates == infoState).astype(float)


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


def traverse(cards, history, traversingPlayer, time, valueBuffer, strategyBuffer):
    plays = len(history)
    player = plays % 2
    infoSet = str(cards[player]) + history
    strategy = getStrategy(infoSet)

    if(IsTerminal(cards, history) is not None):
        return IsTerminal(cards, history)
    elif player == traversingPlayer:
        nodeUtil = 0
        util = [0 for _ in range(NUM_ACTIONS)]
        for a in range(NUM_ACTIONS):
            nextHistory = history + ('p' if a == PASS else 'b')
            util[a] = traverse(cards, nextHistory, traversingPlayer, time, valueBuffer, strategyBuffer)
            print(strategy, util, a)
            nodeUtil += util[a] * strategy[0][a]
        for a in range(NUM_ACTIONS):
            regret = util[a] - nodeUtil
            valueBuffer.insert((infoSet, time, regret))
    else:
        strategyBuffer.insert((infoSet, time, strategy))
        a = getSample(strategy)
        nextHistory = history + ('p' if a == PASS else 'b')
        return traverse(cards, nextHistory, traversingPlayer, time, valueBuffer, strategyBuffer)


def train(iterations):
    print("\n\n\n")
    util = 0
    for i in range(iterations):
        cards = [1, 2, 3]
        random.shuffle(cards)
        valBuffer = ReplayBuffer(1000)
        stratBuffer = ReplayBuffer(1000)
        for traversingPlayer in range(NUM_PLAYERS): 
            traverse(cards, '', traversingPlayer, i, valBuffer, stratBuffer)

train(100)

