import ensemble
import numpy as np
import random


def array2Dict(data, header):
    out = [None] * len(data)

    for i, d in enumerate(data):
        out[i] = {}
        for j, label in enumerate(header):
            try: val = int(d[j])
            except ValueError: val = d[j]
            out[i][label] = val

    return out

def error(pred: list, target: list):
    assert len(pred) == len(target)
    mistakes = 0
    for i in range(len(pred)):
        if pred[i] != target[i]: mistakes += 1
    return mistakes / len(pred)

def str2Num(value):
    if value == 'no': return 0
    else: return 1

if __name__ == '__main__':
    bank_raw_train = [["age", "job", "marital", "education", 
              "default", "balance", "housing", "loan", 
              "contact", "day", "month", "duration", 
              "campaign", "pdays", "previous", "poutcome", "label"]]

    with open("bank-2\\train.csv", "r") as f:
        for line in f:
            terms = line.strip().split(",")
            bank_raw_train.append(terms)

    bank_raw_test = []
    with open("bank-2\\test.csv", "r") as f:
        for line in f:
            terms = line.strip().split(",")
            bank_raw_test.append(terms)

    bank = np.array(array2Dict(bank_raw_train[1:], bank_raw_train[0]))
    test_bank = np.array(array2Dict(bank_raw_test, bank_raw_train[0]))
    idx = list(range(len(bank)))

    ### BAGGED TREES ==============================================================================
    bagged_trees = []
    single_trees = []

    for i in range(100):
        print(i)
        random.shuffle(idx)
        train_bank = bank[idx[:1000]]
        bag = ensemble.BaggedTrees()
        bag.train(train_bank, num_trees=500, num_samples=500)
        bagged_trees.append(bag)
        single_trees.append(bag.getFirstTree())

    bias_single, bias_bagged, var_single, var_bagged = [], [], [], []
    for d in test_bank:
        bagged = list(map(lambda t : str2Num(t.predict([d])[0]), bagged_trees))
        single = list(map(lambda t : str2Num(t.predict(d)), single_trees))
        lab = str2Num(d['label'])

        bias_single.append((lab - np.mean(single)) ** 2)
        bias_bagged.append((lab - np.mean(bagged)) ** 2)
        var_single.append(np.std(single) ** 2)
        var_bagged.append(np.std(bagged) ** 2)

    print(f"single tree: \n    bias: {np.mean(bias_single)}\n    variance: {np.mean(var_single)}\n    GSE: {np.mean(bias_single) + np.mean(var_single)}")
    print(f"bagged trees: \n    bias: {np.mean(bias_bagged)}\n    variance: {np.mean(var_bagged)}\n    GSE: {np.mean(bias_bagged) + np.mean(var_bagged)}")


    ### RANDOM FOREST ==============================================================================
    randomforests = []
    single_trees = []
    for i in range(50):
        random.shuffle(idx)
        train_bank = bank[idx[:1000]]
        rf = ensemble.RandomForest()
        rf.train(train_bank, num_trees=500, num_samples=500)
        randomforests.append(rf)
        single_trees.append(rf.getFirstTree())

    bias_single, bias_randfor, var_single, var_randfor = [], [], [], []
    for d in test_bank:
        randfor = list(map(lambda t : str2Num(t.predict([d])[0]), randomforests))
        single = list(map(lambda t : str2Num(t.predict(d)), single_trees))
        lab = str2Num(d['label'])

        bias_single.append((lab - np.mean(single)) ** 2)
        bias_randfor.append((lab - np.mean(randfor)) ** 2)
        var_single.append(np.std(single) ** 2)
        var_randfor.append(np.std(randfor) ** 2)

    print(f"single tree: \n    bias: {np.mean(bias_single)}\n    variance: {np.mean(var_single)}\n    GSE: {np.mean(bias_single) + np.mean(var_single)}")
    print(f"random forest: \n    bias: {np.mean(bias_randfor)}\n    variance: {np.mean(var_randfor)}\n    GSE: {np.mean(bias_randfor) + np.mean(var_randfor)}")
