# %%
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from  buffer import ReplayBuffer

# %%
PASS = 0
BET = 1
NUM_ACTIONS = 2
NUM_PLAYERS = 2
nodeMap = {}


def getStrategy(infoSet, regretNet):
    encoder = KuhnEncoder()
    regrets = tf.reshape(regretNet(encoder.encode(infoSet)), (NUM_ACTIONS,))
    regrets = tf.nn.relu(regrets)
    strategy, regretSum = tf.linalg.normalize(regrets, ord=1)
    if regretSum < 0.01:
        strategy, _ = tf.linalg.normalize(np.ones(tf.shape(strategy), dtype=np.float32), ord=1)
    return strategy
    

def getAverageStrategy(infoSet, strategyNet):
    encoder = KuhnEncoder()
    regrets = tf.reshape(strategyNet(encoder.encode(infoSet)), (NUM_ACTIONS,))
    regret_sum = tf.sum(regrets)
    

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
    

# %%
def generateDataset(buffer, validation_fraction=3/4):
    x, y = [], []
    
    splitIdx = int(buffer.size * validation_fraction)
    for (i, t, sigma) in buffer.data:
        x.append(KuhnEncoder().encode(i))
        y.append(sigma)
    x, y = np.array(x), np.array(y)
    return (x[:splitIdx], y[:splitIdx]), (x[splitIdx:], y[splitIdx:])




# %%
def traverse(cards, history, traversingPlayer, time, valueBuffer, strategyBuffer, regretNet):
    plays = len(history)
    player = plays % 2
    infoSet = str(cards[player]) + history
    strategy = getStrategy(infoSet, regretNet)

    if(IsTerminal(cards, history, traversingPlayer) is not None):
        return IsTerminal(cards, history, traversingPlayer)
    elif player == traversingPlayer:
        nodeUtil = 0
        util = [0 for _ in range(NUM_ACTIONS)]
        regret = [0 for _ in range(NUM_ACTIONS)]
        for a in range(NUM_ACTIONS):
            nextHistory = history + ('p' if a == PASS else 'b')
            util[a] = traverse(cards, nextHistory, traversingPlayer, time, 
                valueBuffer, strategyBuffer, regretNet)
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
                        valueBuffer, strategyBuffer, regretNet)


def train(inner_iterations, outer_iterations):
    stratBuffer = ReplayBuffer(30)
    regretNet = keras.Sequential([
        keras.layers.Normalization(input_shape=[12, ]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(2)
    ])
    strategyNet = keras.Sequential([
        keras.layers.Normalization(input_shape=[12, ]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(2, activation="sigmoid")
    ])

    regretNet.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.MeanSquaredError(),
        metrics=keras.losses.MeanSquaredError(),
    )
    
    strategyNet.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.MeanSquaredError(),
    )

    
    
    for i in range(outer_iterations):
        valBuffer = ReplayBuffer(30)
        for j in range(inner_iterations):
            cards = [1, 2, 3]
            random.shuffle(cards)
            
            for traversingPlayer in range(NUM_PLAYERS): 
                traverse(cards, '', traversingPlayer, j, valBuffer, stratBuffer, regretNet)
        
        (x_train, y_train), (x_val, y_val) = generateDataset(valBuffer)
        regretNet.fit(
            x_train,
            y_train,
            batch_size=8,
            epochs=5,
            validation_data=(x_val,y_val)
        )
    
    print("Finally, training the strategy.")
    (x_train, y_train), (x_val, y_val) = generateDataset(stratBuffer)
    strategyNet.fit(
        x_train,
        y_train,
        batch_size=8,
        epochs=1,
        validation_data=(x_val,y_val)
    )
        
    # train strategy net
    return valBuffer, stratBuffer

# %%
_, buffer = train(50, 1)

# %%
