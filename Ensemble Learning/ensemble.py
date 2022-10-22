from DecisionTree import DecisionTree, RandomForestTree
import numpy as np
import multiprocessing as mp
from math import log, exp
from statistics import mode
import random


def bagAndMakeTree(data, num_samples):
    bag = []
    for _ in range(num_samples):
        x = random.randrange(0, len(data))
        bag.append(data[x])

    tree = DecisionTree()
    tree.makeTree(bag)
    return tree

class BaggedTrees:
    def __init__(self):
        self.trees = list

    def train(self, data: list, num_trees: int = 100, num_samples: int = 1000, num_workers = None):
        mult_data = [data] * num_trees
        mult_samp = [num_samples] * num_trees

        with mp.Pool(num_workers) as pool:
            self.trees = pool.starmap(bagAndMakeTree, zip(mult_data, mult_samp))

    def getFirstTree(self):
        return self.trees[0]

    def predict(self, data, num_workers = 4):
        pred = np.zeros_like(data)

        for i, d in enumerate(data):
            pred[i] = mode(map(lambda tree : tree.predict(d), self.trees))

        return pred

def rfBagTree(data, num_samples, num_attributes):
    bag = []
    for _ in range(num_samples):
        x = random.randrange(0, len(data))
        bag.append(data[x])

    tree = RandomForestTree()
    tree.makeTree(bag, num_attributes=num_attributes)
    return tree

class RandomForest:
    def __init__(self):
        self.trees = list

    def train(self, data: list, num_trees: int = 100, num_samples: int = 1000, num_attributes: int = 4, num_workers = None):

        mult_data = [data] * num_trees
        mult_samp = [num_samples] * num_trees
        mult_attr = [num_attributes] * num_trees

        with mp.Pool(num_workers) as pool:
            self.trees = pool.starmap(rfBagTree, zip(mult_data, mult_samp, mult_attr))
            
    def getFirstTree(self):
        return self.trees[0]

    def predict(self, data, num_workers = 4):
        pred = np.zeros_like(data)

        for i, d in enumerate(data):
            pred[i] = mode(map(lambda tree : tree.predict(d), self.trees))

        return pred