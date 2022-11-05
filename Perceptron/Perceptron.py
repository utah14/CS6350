import numpy as np
from os import makedirs
import csv

# the standard Perceptron
class Perceptron:
    def __init__(self):
        self.weights = np.ndarray
        
    def __init__(self, X, y, r:float = 0.001, epochs: int=10):
        self.weights = np.ndarray
        self.train(X, y, r, epochs)

    def append_bias(self, X):
        return np.insert(X, 0, [1]*len(X), axis=1)

    def train(self, X, y, r:float=0.001, epochs: int=10):
        X = self.append_bias(X)
        self.weights = np.zeros_like(X[0])

        for e in range(epochs):
            idxs = np.arange(len(X))
            np.random.shuffle(idxs)
            for i in idxs:
                if y[i] * np.dot(self.weights, X[i]) <= 0:
                    self.weights += r*(y[i]*X[i])
    
    def predict(self, X) -> np.ndarray:
        X = self.append_bias(X)
        pred = lambda d : np.sign(np.dot(self.weights, d))
        return np.array([pred(xi) for xi in X])

class Voted(Perceptron):
    def __init__(self, X, y, r:float = 1e-3, epochs: int=10):
        self.votes = np.ndarray
        self.train(X, y, r, epochs)

    def train(self, X, y, r:float=1e-3, epochs: int=10):
        X = self.append_bias(X)
        m = 0
        weights = [np.zeros_like(X[0])]
        cm = [0]

        for ep in range(epochs):
            index = np.arange(len(X))
            np.random.shuffle(index)
            for i in index:
                if y[i] * np.dot(weights[m], X[i]) <= 0:
                    weights[m] += r*(y[i]*X[i])
                    weights.append(weights[m].copy())
                    m += 1
                    cm.append(1)
                else: cm[m] += 1

        self.votes = np.array(list(zip(weights, cm)), dtype=object)
    
    def predict(self, X) -> np.ndarray:
        X = self.append_bias(X)
        preds = np.zeros(len(X), dtype=int)
        for i in range(len(preds)):
            inner = 0
            for w, c in self.votes:
                inner += c * np.sign(np.dot(w, X[i]))
            preds[i] = np.sign(inner)
        return preds

class Averaged(Perceptron):
    def train(self, X, y, r:float=1e-3, epochs: int=10):
        X = self.append_bias(X)
        self.weights = np.zeros_like(X[0])
        weights = np.zeros_like(X[0])

        for e in range(epochs):
            index = np.arange(len(X))
            np.random.shuffle(index)
            for i in index:
                if y[i] * np.dot(weights, X[i]) <= 0:
                    weights += r*(y[i]*X[i])
                self.weights = self.weights + weights

dataset_loc = "bank-note/bank-note/"

X_train = []
y_train = []
with open(dataset_loc + "train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        X_train.append(terms_flt[:-1])
        y_train.append(-1 if terms_flt[-1] == 0 else 1)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []
with open(dataset_loc + "test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        X_test.append(terms_flt[:-1])
        y_test.append(-1 if terms_flt[-1] == 0 else 1)

X_test = np.array(X_test)
y_test = np.array(y_test)

print("**** Standard Perceptron ****")
sp = Perceptron(X_train, y_train, r=0.1)
print(f"learned weights: {sp.weights}")
print(f"training error: {1-np.mean(y_train == sp.predict(X_train))}")
print(f"testing error: {1-np.mean(y_test == sp.predict(X_test))}")

print("**** Voted Perceptron ****")
vp = Voted(X_train, y_train, r=0.1)
print(f"learned weights and counts: {vp.votes}")
with open('VotedPeceptron_weights.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['b', 'x1', 'x2', 'x3', 'x4', 'Cm'])
    for w in vp.votes:
        row = w[0]
        row = np.append(row, w[1])
        writer.writerow(row)
print(f"training error: {1-np.mean(y_train == vp.predict(X_train))}")
print(f"testing error: {1-np.mean(y_test == vp.predict(X_test))}")

print("**** Averaged Perceptron ****")
ap = Averaged(X_train, y_train, r=0.1)
print(f"learned weights: {ap.weights}")
print(f"training error: {1-np.mean(y_train == ap.predict(X_train))}")
print(f"testing error: {1-np.mean(y_test == ap.predict(X_test))}")
