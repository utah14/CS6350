from os import makedirs
import ensemble
import numpy as np

import matplotlib.pyplot as plt

def error(pred: list, target: list):
    assert len(pred) == len(target)
    mistakes = 0
    for i in range(len(pred)):
        if pred[i] != target[i]: mistakes += 1
    return mistakes / len(pred)

def HandleLine(line):
    terms = line.strip().split(",")
    t_dict = { # TODO: better way of doing this?
        "age": int(terms[0]), # numeric
        "job": terms[1], # categorical
        "marital": terms[2], # categorical
        "education": terms[3], # categorical
        "default": terms[4], # binary
        "balance": int(terms[5]), #numeric
        "housing": terms[6], # binary
        "loan": terms[7], # binary
        "contact": terms[8], # categorical
        "day": int(terms[9]), # numeric
        "month": terms[10], # categorical
        "duration": int(terms[11]), # numeric
        "campaign": int(terms[12]), # numeric
        "pdays": int(terms[13]), # numeric
        "previous": int(terms[14]), # numeric 
        "poutcome": terms[15], # categorical

        "label": terms[16] # binary
    }
    return t_dict

if __name__ == '__main__':
    train_bank = []
    with open("bank-2\\train.csv", "r") as f:
        for line in f:
            train_bank.append(HandleLine(line))

    test_bank = []
    with open("bank-2\\test.csv", "r") as f:
        for line in f:
            test_bank.append(HandleLine(line))

    print("datasets loaded")

    print("Bagged trees...")
    x_pts = list(range(1,30,2)) + list(range(30,100,10)) + list(range(100, 550, 50))
    train_err = []
    test_err = []

    for x in x_pts:
        print(f"# trees: {x}")
        
        bag = ensemble.BaggedTrees()
        bag.train(train_bank, num_trees=x, num_samples=1000)

        train_pred = bag.predict(train_bank)
        train_err.append(error(train_pred, [d['label'] for d in train_bank]))

        test_pred = bag.predict(test_bank)
        test_err.append(error(test_pred, [d['label'] for d in test_bank]))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_pts, train_err, color = 'tab:blue', label = "training")
    ax.plot(x_pts, test_err, color = 'tab:orange', label = "testing")
    ax.legend()
    ax.set_title("Bagged Trees")
    ax.set_xlabel("The count of trees")
    ax.set_ylabel("Error")

    plt.savefig("bagged_bank.png")
    plt.clf()

    print("Random forest")
    train_err_2 = []
    train_err_4 = []
    train_err_6 = []
    test_err_2 = []
    test_err_4 = []
    test_err_6 = []

    for x in x_pts:
        print(f"# trees: {x}")

        rf_2 = ensemble.RandomForest()
        rf_2.train(train_bank, num_trees=x, num_samples=1000, num_attributes=2)

        train_pred = rf_2.predict(train_bank)
        train_err_2.append(error(train_pred, [d['label'] for d in train_bank]))

        test_pred = rf_2.predict(test_bank)
        test_err_2.append(error(test_pred, [d['label'] for d in test_bank]))

        rf_4 = ensemble.RandomForest()
        rf_4.train(train_bank, num_trees=x, num_samples=1000)

        train_pred = rf_4.predict(train_bank)
        train_err_4.append(error(train_pred, [d['label'] for d in train_bank]))

        test_pred = rf_4.predict(test_bank)
        test_err_4.append(error(test_pred, [d['label'] for d in test_bank]))

        rf_6 = ensemble.RandomForest()
        rf_6.train(train_bank, num_trees=x, num_samples=1000, num_attributes=6)

        train_pred = rf_6.predict(train_bank)
        train_err_6.append(error(train_pred, [d['label'] for d in train_bank]))

        test_pred = rf_6.predict(test_bank)
        test_err_6.append(error(test_pred, [d['label'] for d in test_bank]))

        

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_pts, train_err_2, label = "training, Subset = 2")
    ax.plot(x_pts, test_err_2, label = "testing, Subset = 2")
    ax.plot(x_pts, train_err_4, label = "training, Subset = 4")
    ax.plot(x_pts, test_err_4, label = "testing, Subset = 4")
    ax.plot(x_pts, train_err_6, label = "training, Subset = 6")
    ax.plot(x_pts, test_err_6, label = "testing, Subset = 6")
    ax.legend()
    ax.set_title("Random Forest")
    ax.set_xlabel("the count of trees")
    ax.set_ylabel("Error")

    plt.savefig("randomforest_bank.png")
    plt.clf()