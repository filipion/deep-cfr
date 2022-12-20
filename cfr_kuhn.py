import random

PASS = 0
BET = 1
NUM_ACTIONS = 2
nodeMap = {}

class KuhnNode():

    def __init__(self, infoSet):
        self.infoSet = infoSet
        self.regretSum = [0 for _ in range(NUM_ACTIONS)]
        self.strategy = [0 for _ in range(NUM_ACTIONS)]
        self.strategySum = [0 for _ in range(NUM_ACTIONS)]

    def __repr__(self):
        return "CFR node for the infoset {infoset}. Strategy: {strategy}".format(infoset=self.infoSet, 
            strategy=["{:.2f}".format(x) for x in self.getAverageStrategy()])

    def getStrategy(self, realizationWeight):
        normalizingSum = 0
        for a in range(NUM_ACTIONS):
            self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
            normalizingSum += self.strategy[a]

        for a in range(NUM_ACTIONS):
            if(normalizingSum > 0):
                self.strategy[a] /= normalizingSum
            else:
                self.strategy[a] = 1 / NUM_ACTIONS
            self.strategySum[a] += realizationWeight * self.strategy[a]

        return self.strategy

    def getAverageStrategy(self):
        averageStrategy = [0 for _ in range(NUM_ACTIONS)]
        normalizingSum = 0
        for a in range(NUM_ACTIONS):
            normalizingSum += self.strategySum[a]

        for a in range(NUM_ACTIONS):
            if normalizingSum == 0:
                averageStrategy[a] = 1 / NUM_ACTIONS
            else:
                averageStrategy[a] = self.strategySum[a] / normalizingSum

        return averageStrategy

def cfr(cards, history, p0, p1, time):
    plays = len(history)
    player = plays % 2
    opponent = 1 - player
    if player == 0:
        realizationWeight = p0
    else:
        realizationWeight = p1

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

def train(iterations):
    util = 0
    for i in range(iterations):
        cards = [1, 2, 3]
        random.shuffle(cards)
        util += cfr(cards, '', 1.0, 1.0, i)
        if(i % (iterations // 10) == 0):
            print("Iteration number: {}/{}".format(i, iterations))
            logInfosets()
            print("Avg, utility for player 1: {:.4f}".format(util / (i + 1)))

def logInfosets():
    infoSets = sorted(nodeMap.keys())
    for node in infoSets:
        print(nodeMap[node])

iterations = 1000000
train(iterations)
    