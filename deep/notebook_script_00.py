
# %%
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from  buffer import ReplayBuffer
import matplotlib.pyplot as plt
# %%
PASS = 0
BET = 1
NUM_ACTIONS = 2
NUM_PLAYERS = 2
INFOSTATES = ["1", "2", "3", "1p", "2p", "3p", "1b", "2b", "3b", "1pb", "2pb", "3pb"]
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
    strategy = tf.reshape(strategyNet(encoder.encode(infoSet)), (NUM_ACTIONS,))
    strategy, _ = tf.linalg.normalize(strategy, ord=1)
    return strategy
    

def getSample(strategy):
    cum_probs = tf.cumsum(strategy)
    rand_num = tf.random.uniform([], 0, 1)
    i = tf.argmax(cum_probs > rand_num)
    return i


class KuhnEncoder():
    def __init__(self):
        self.infostates = pd.Series(INFOSTATES)
    
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
                return -1 if plays % 2 != player else 1
        elif doubleBet:
            return 2 if isPlayerCardHigher else -2
    else:
        return None
    

# %%
def generateDataset(buffer, validation_fraction=3/4):
    x, y = [], []
    random.shuffle(buffer.data)
    for (i, t, sigma) in buffer.data:
        x.append(KuhnEncoder().encode(i))
        y.append(sigma)
    x, y = np.array(x), np.array(y)
    return x, y

def plot_metric(history, metric, network_name):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.title('Training and validation {}: {}'.format(metric, network_name))
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()




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


# %%
def train(inner_iterations, outer_iterations, strategyNet, logger):
    stratBuffer = ReplayBuffer(400)
    valBuffer = ReplayBuffer(100)
    
    regretNet = keras.Sequential([
        keras.layers.Normalization(input_shape=[12, ]),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(2)
    ])
    regretNet.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.MeanSquaredError(),
        metrics=keras.losses.MeanSquaredError(),
    )
    
    for _ in range(outer_iterations):
        for j in range(inner_iterations):
            cards = [1, 2, 3]
            random.shuffle(cards)
            
            for traversingPlayer in range(NUM_PLAYERS): 
                traverse(cards, '', traversingPlayer, j, valBuffer, stratBuffer, regretNet)
        
        x_train, y_train = generateDataset(valBuffer)
        history = regretNet.fit(
            x_train,
            y_train,
            batch_size=16,
            epochs=20,
            validation_split=0.5
            #callbacks=[tf.keras.callbacks.EarlyStopping()]
        )
        plot_metric(history, 'loss', 'Value Net')
    
    print("Finally, training the strategy.")
    x_train, y_train = generateDataset(stratBuffer)
    print(stratBuffer.data)
    history = strategyNet.fit(
        x_train,
        y_train,
        batch_size=16,
        epochs=100,
        validation_split=0.25,
        # callbacks=[tf.keras.callbacks.EarlyStopping()]
    )
    plot_metric(history, 'loss', 'Strategy Net')
        
    # train strategy net
    return valBuffer, stratBuffer, strategyNet, logger

# %%
strategyNet = keras.Sequential([
    keras.layers.Normalization(input_shape=[12, ]),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="sigmoid")
])

strategyNet.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.MeanSquaredError(),
)

untrained = keras.Sequential([
    keras.layers.Normalization(input_shape=[12, ]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(2, activation="sigmoid")
])

untrained.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.MeanSquaredError(),
)

# %%
def simulate_game(agents, result_player):
    cards = [1, 2, 3]
    random.shuffle(cards)
    history = ''
    player = 0
    while(IsTerminal(cards, history, player) is None):
        infoSet = str(cards[player]) + history
        strategy = getAverageStrategy(infoSet, agents[player])
        a = getSample(strategy)
        history = history + ('p' if a == PASS else 'b')
        player = 1 - player
    
    return IsTerminal(cards, history, result_player)

def rollout(agents, num_rollouts, result_player):
    victories = 0
    for i in range(num_rollouts):
        if i % 10 == 0:
            print("simulating {}th game".format(i), flush=True)
        if simulate_game(agents, result_player) > 0:
            victories += 1
    return victories

def symmetric_rollout(agents, num_rollouts=50):
    return rollout(agents, num_rollouts, 0) + rollout(agents, num_rollouts, 1)


# %%
logger = []
_, strat_buffer, _, logger = train(100, 10, strategyNet, logger)


# %%
def plot_strategy(strategyNet):
    for i in INFOSTATES:
        strategy = getAverageStrategy(i, strategyNet)
        print(i, strategy)

plot_strategy(strategyNet)

# %%
print(rollout([untrained, untrained], 1000, 0))
print(rollout([strategyNet, untrained], 1000, 0))