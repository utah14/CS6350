import math
import numpy as np
from statistics import mode
import random

class TreeNode(object):
    def __init__(self, nodetype = None, attr = None, value = None, finalclass = None):
        self.type = nodetype
        self.attr = attr
        self.value = value
        self.finalclass = finalclass
        self.children = []

    def toJSON(self):
        dict = {
            "type": self.type,
            "attr": self.attr,
            "value": self.value,
            "finalclass": self.finalclass,
            "children": []
        }

        for c in self.children:
            dict["children"].append(c.toJSON())

        return dict

def mostCommon(data, attribute = "label"):
    values = list(filter(lambda x: x != "unknown", [d[attribute] for d in data]))
    return mode(values)

def splitAtMedian(data, attribute):
    values = [d[attribute] for d, w in data]
    median = np.median(values)
    lower = []
    upper = []

    for d, w in data:
        if d[attribute] < median: lower.append((d, w))
        else: upper.append((d, w))

    return lower, upper, median

def GiniIndex(data: list):
    counter = {}
    weight_sum = np.sum([w for d, w in data])

    for d, w in data:
        if counter.get(d["label"]) == None:  counter[d["label"]] = w
        else: counter[d["label"]] += w
    
    gini = 0
    for v in counter.values():
        # gini += (v / len(data))**2
        gini += (v / weight_sum)**2

    return 1 - gini

def InformationGain(data: list, attribute: str, purity = GiniIndex):
    gain = 0
    weight_sum = np.sum([w for d, w in data])
    if type(data[0][0][attribute]) == str:
        unique_vals = np.unique(np.array([d[attribute] for d, w in data]))
        for val in unique_vals:
            subset = []
            for d, w in data:
                if d[attribute] == val:
                    subset.append((d, w))
            # gain += (len(subset) / len(data)) * purity(subset)
            gain += (np.sum([w for d, w in subset]) / weight_sum) * purity(subset)
        
    elif type(data[0][0][attribute] == int):
        lower, upper, _ = splitAtMedian(data, attribute)
        # gain = ( (len(lower) / len(data)) * purity(lower) ) + ( (len(upper) / len(data)) * purity(upper) )
        gain = ((np.sum([w for d, w in lower]) / weight_sum) * purity(lower)) + ((np.sum([w for d, w in upper]) / weight_sum) * purity(upper))
        
    return(purity(data) - gain)

def allSame(data):
        return len(np.unique(np.array([d["label"] for d in data]))) == 1

class DecisionTree:
    def __init__(self, purity_function = InformationGain, ):
        self.root = TreeNode(nodetype="root")
        self.purity_function = purity_function
        self.max_depth = 9999
        self.mostLabel = "na"

    # public makeTree starter function
    def makeTree(self, data: list, weights = None, max_depth: int = None):
        if max_depth != None: self.max_depth = max_depth
        if weights == None: weights = [1/len(data)]*len(data)
        self.mostLabel = mostCommon(data)
        self.root = self._makeTree(data, weights, self.root, 0, ["label"])

    # private recursive _makeTree function
    def _makeTree(self, data: list, weights: list, node, depth, used_attrs: list):
        # base cases
        if len(data) == 0: # if the set of data is empty,
            node.type = "leaf"
            node.finalclass = self.mostLabel
            return node # return a node with the most common label
        if allSame(data): # if the data all have the same label
            node.type = "leaf"
            node.finalclass = data[0]["label"]
            return node # # return a node with that label
        if depth >= self.max_depth: # if the max depth has been met, 
            node.type = "leaf"
            node.finalclass = mostCommon(data)
            return node # return a node with the most common label

        # find best split given purity function
        max = { "val": -np.inf, "attr": "none_found" }
        for attr in data[0].keys():
            if attr in used_attrs:
                continue
            purity = self.purity_function(list(zip(data, weights)), attr)
            # print(f"attr: {attr}, gain: {purity}")
            if purity > max["val"]:
                max["val"] = purity
                max["attr"] = attr
        
        new_attrs = used_attrs.copy()
        new_attrs.append(max["attr"])

        # if we have exhausted all attributes but still not perfectly partitioned the data, assign most common label
        if max["attr"] == "none_found":
            node.type = "leaf"
            node.finalclass = mostCommon(data)
            return node

        # for every unique value of the best split attribute, make a new child node
        if type(data[0][max["attr"]]) == str:
            unique_vals = np.unique(np.array([d[max["attr"]] for d in data]))
            for val in unique_vals:
                childNode = TreeNode(nodetype="split", attr=max["attr"], value=val)
                new_data = [d for d, w in zip(data, weights) if d[max["attr"]] == val]
                new_weights = [w for d, w in zip(data, weights) if d[max["attr"]] == val]
                node.children.append(self._makeTree(new_data, new_weights, childNode, depth+1, new_attrs))

        elif type(data[0][max["attr"]]) == int:
            lower, upper, median = splitAtMedian(list(zip(data, weights)), max["attr"])

            lower_data = [d for d, w in lower]
            upper_data = [d for d, w in upper]

            lower_weights = [w for d, w in lower]
            upper_weights = [w for d, w in upper]

            child_lower = TreeNode(nodetype="split", attr=max["attr"], value=(-np.inf, median))
            child_upper = TreeNode(nodetype="split", attr=max["attr"], value=(median, np.inf))

            node.children.append(self._makeTree(lower_data, lower_weights, child_lower, depth+1, new_attrs))
            node.children.append(self._makeTree(upper_data, upper_weights, child_upper, depth+1, new_attrs))
        return node
    
    # exports tree in JSON format
    def toJSON(self): return self.root.toJSON()

    # predicts label based on attributes
    def predict(self, value): return self._predict(value, self.root)

    def _predict(self, value, node):
        if node.type == "leaf":
            return node.finalclass
        
        for child in node.children:
            attr = child.attr
            if type(value[attr]) == str:
                if value[attr] == child.value:
                    return self._predict(value, child)
            elif type(value[attr]) == int:
                if (value[attr] >= child.value[0]) & (value[attr] < child.value[1]):
                    return self._predict(value, child)

class RandomForestTree:
    def __init__(self, purity_function = InformationGain):
        self.root = TreeNode(nodetype="root")
        self.purity_function = purity_function
        self.max_depth = 9999
        self.mostLabel = "na"

    # public makeTree starter function
    def makeTree(self, data: list, num_attributes: int = 4, max_depth: int = None):
        if max_depth != None: self.max_depth = max_depth
        weights = [1/len(data)]*len(data)
        self.mostLabel = mostCommon(data)
        self.root = self._makeTree(data, weights, num_attributes, self.root, 0, ["label"])

    # private recursive _makeTree function
    def _makeTree(self, data: list, weights: list, num_attributes, node, depth, used_attrs: list):
        # base cases
        if len(data) == 0: # if the set of data is empty,
            node.type = "leaf"
            node.finalclass = self.mostLabel
            return node # return a node with the most common label
        if allSame(data): # if the data all have the same label
            node.type = "leaf"
            node.finalclass = data[0]["label"]
            return node # # return a node with that label
        if depth >= self.max_depth: # if the max depth has been met, 
            node.type = "leaf"
            node.finalclass = mostCommon(data)
            return node # return a node with the most common label

        # find best split given purity function
        max = { "val": -np.inf, "attr": "none_found" }
        A = list(data[0].keys())
        G = []
        if len(A)-len(used_attrs) <= num_attributes: G = A
        else:
            i=0
            while i < num_attributes:
                idx = random.randrange(0, len(A))
                if A[idx] not in G: 
                    G.append(A[idx])
                    i += 1
            
        for attr in G:
            if attr in used_attrs:
                continue
            purity = self.purity_function(list(zip(data, weights)), attr)
            # print(f"attr: {attr}, gain: {purity}")
            if purity > max["val"]:
                max["val"] = purity
                max["attr"] = attr
        
        new_attrs = used_attrs.copy()
        new_attrs.append(max["attr"])

        # if we have exhausted all attributes but still not perfectly partitioned the data, assign most common label
        if max["attr"] == "none_found":
            node.type = "leaf"
            node.finalclass = mostCommon(data)
            return node

        # for every unique value of the best split attribute, make a new child node
        if type(data[0][max["attr"]]) == str:
            unique_vals = np.unique(np.array([d[max["attr"]] for d in data]))
            for val in unique_vals:
                childNode = TreeNode(nodetype="split", attr=max["attr"], value=val)
                new_data = [d for d, w in zip(data, weights) if d[max["attr"]] == val]
                new_weights = [w for d, w in zip(data, weights) if d[max["attr"]] == val]
                node.children.append(self._makeTree(new_data, new_weights, num_attributes, childNode, depth+1, new_attrs))

        elif type(data[0][max["attr"]]) == int:
            lower, upper, median = splitAtMedian(list(zip(data, weights)), max["attr"])

            lower_data = [d for d, w in lower]
            upper_data = [d for d, w in upper]

            lower_weights = [w for d, w in lower]
            upper_weights = [w for d, w in upper]

            child_lower = TreeNode(nodetype="split", attr=max["attr"], value=(-np.inf, median))
            child_upper = TreeNode(nodetype="split", attr=max["attr"], value=(median, np.inf))

            node.children.append(self._makeTree(lower_data, lower_weights, num_attributes, child_lower, depth+1, new_attrs))
            node.children.append(self._makeTree(upper_data, upper_weights, num_attributes, child_upper, depth+1, new_attrs))
        return node
    
    # exports tree in JSON format
    def toJSON(self): return self.root.toJSON()

    # predicts label based on attributes
    def predict(self, value): return self._predict(value, self.root)

    def _predict(self, value, node):
        if node.type == "leaf":
            return node.finalclass
        
        for child in node.children:
            attr = child.attr
            if type(value[attr]) == str:
                if value[attr] == child.value:
                    return self._predict(value, child)
            elif type(value[attr]) == int:
                if (value[attr] >= child.value[0]) & (value[attr] < child.value[1]):
                    return self._predict(value, child)
