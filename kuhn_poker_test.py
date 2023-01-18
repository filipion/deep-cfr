from cfr_kuhn import *

epsilon = 1e-2

def testKuhnPoker():
    iterations = 400000
    train(iterations)
    strategy = {key : val.getAverageStrategy() for key, val in nodeMap.items()}
    testKuhnStrategy(strategy)

def testVectorEquality(v, w):
    for x, y in zip(v, w):
        assert x < y + epsilon and x > y - epsilon

def testKuhnStrategy(strategy):
    testVectorEquality(strategy['3p'], [0, 1])
    testVectorEquality(strategy['3b'], [0, 1])
    testVectorEquality(strategy['2p'], [1, 0])
    testVectorEquality(strategy['2b'], [2/3, 1/3])
    testVectorEquality(strategy['1p'], [2/3, 1/3])
    testVectorEquality(strategy['1b'], [1, 0])

    alpha = strategy['1'][1]
    testVectorEquality(strategy['1'], [1 - alpha, alpha])
    testVectorEquality(strategy['2'], [1, 0])
    testVectorEquality(strategy['3'], [1 - 3 * alpha, 3 * alpha])
    testVectorEquality(strategy['1pb'], [1, 0])
    testVectorEquality(strategy['2pb'], [2/3 - alpha, 1/3 + alpha])
    testVectorEquality(strategy['3pb'], [0, 1])


testKuhnPoker()