import random
import tensorflow as tf
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
    keras.layers.Dense(2, activation="sigmoid")
])

regretNet.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

strategyNet = keras.models.clone_model(regretNet)

def getStrategy(infoSet):
    encoder = KuhnEncoder()
    return tf.reshape(regretNet(encoder.encode(infoSet)), (NUM_ACTIONS,))

def getAverageStrategy(infoSet):
    encoder = KuhnEncoder()
    return tf.reshape(strategyNet(encoder.encode(infoSet)), (NUM_ACTIONS,))

def getSample(strategy):
    cum_probs = tf.cumsum(strategy)
    rand_num = tf.random.uniform([], 0, 1)
    i = tf.argmax(cum_probs > rand_num)
    return i


class KuhnEncoder():
    def __init__(self):
        self.infostates = pd.Series(["1", "2", "3", "1p", "2p", "3p", 
                                     "1b", "2b", "3b", "1pb", "2pb", "3pb"])
    
    def encode(self, infoState):
        return np.array(self.infostates == infoState).astype(float)


def IsTerminal(cards, history, player):
    plays = len(history)
    opponent = 1 - player
    if plays > 1 :
        terminalPass = history[-1] == "p"
        doubleBet = history[-2:] == "bb"
        isPlayerCardHigher = cards[player] > cards[opponent]
        if terminalPass:
            if history == "pp":
                return 1 if isPlayerCardHigher else -1
            else:
                return 1
        elif doubleBet:
            return 2 if isPlayerCardHigher else -2
    else:
        return None


def traverse(cards, history, traversingPlayer, time, valueBuffer, strategyBuffer):
    plays = len(history)
    player = plays % 2
    infoSet = str(cards[player]) + history
    strategy = getStrategy(infoSet)

    if(IsTerminal(cards, history, traversingPlayer) is not None):
        return IsTerminal(cards, history, traversingPlayer)
    elif player == traversingPlayer:
        nodeUtil = 0
        util = [0 for _ in range(NUM_ACTIONS)]
        regret = [0 for _ in range(NUM_ACTIONS)]
        for a in range(NUM_ACTIONS):
            nextHistory = history + ('p' if a == PASS else 'b')
            util[a] = traverse(cards, nextHistory, traversingPlayer, time, 
                valueBuffer, strategyBuffer)
            nodeUtil += util[a] * strategy[a]
        for a in range(NUM_ACTIONS):
            regret[a] = util[a] - nodeUtil
            valueBuffer.insert((infoSet, time, regret))
        return nodeUtil
    else:
        strategyBuffer.insert((infoSet, time, strategy))
        a = getSample(strategy)
        nextHistory = history + ('p' if a == PASS else 'b')
        return traverse(cards, nextHistory, traversingPlayer, time, 
                        valueBuffer, strategyBuffer)


def train(inner_iterations, outer_iterations):
    print("\n\n\n")
    for i in range(outer_iterations):
        valBuffer = ReplayBuffer(100)
        stratBuffer = ReplayBuffer(100)
        for j in range(inner_iterations):
            cards = [1, 2, 3]
            random.shuffle(cards)
            
            for traversingPlayer in range(NUM_PLAYERS): 
                traverse(cards, '', traversingPlayer, j, 
                        valBuffer, stratBuffer)
            # train value net
        # train strategy net

    return valBuffer
