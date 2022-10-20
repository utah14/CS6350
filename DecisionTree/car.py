from cmath import nan
from turtle import end_fill
import pandas as pd #for manipulating the csv data
import numpy as np #for mathematical calculation
import math as math

def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0] #the total size of the dataset
    total_entr = 0
    
    for c in class_list: #for each class in the label
        total_class_count = train_data[train_data[label] == c].shape[0] #number of the class
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) #entropy of the class
        if(math.isnan(total_class_entr)):
            total_entr+=0
        else:
            total_entr += total_class_entr #adding the class entropy to the total entropy of the dataset
    return total_entr


def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0] #the total size of the attribute
    entropy = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0] #row count of class c 
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count #probability of the class
            entropy_class = - probability_class * np.log2(probability_class)  #entropy
            if(math.isnan(entropy_class)):
                entropy_class=0
        entropy += entropy_class
    return entropy


def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() #unqiue values of the feature
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list) #calculcating entropy for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value
        
    return calc_total_entropy(train_data, label, class_list) - feature_info #calculating information gain by subtracting


def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) #finding the feature names in the dataset
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  #for each feature in the dataset
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature


def make_tree_Entropy(root, prev_feature_value, train_data, label, class_list,root_feature):
    if train_data.shape[0] != 0: #if dataset becomes enpty after updating
        max_info_feature = find_most_informative_feature(train_data, label, class_list) #most informative feature
        if(root_feature==max_info_feature):
            return

        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) #getting tree node and updated dataset
        next_root = None
        if prev_feature_value != None: #add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: #add to root of the tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        for node, branch in list(next_root.items()): #iterating the tree node
            if branch == "?": #if it is expandable
                feature_value_data = train_data[train_data[max_info_feature] == node] #using the updated dataset
                make_tree_Entropy(next_root, node, feature_value_data, label, class_list,max_info_feature) #recursive call with updated dataset

def id3_Entropy(train_data_m, label):
    train_data = train_data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    class_list = train_data[label].unique() #getting unqiue classes of the label
    make_tree_Entropy(tree, None, train_data_m, label, class_list,"root") # recursion
    return tree

def calc_total_Gini(train_data, label, class_list):
    total_row = train_data.shape[0] #the total size of the dataset
    total_gini = 0
    
    for c in class_list: #for each class in the label
        total_class_count = train_data[train_data[label] == c].shape[0] #number of the class
        total_class_gini = pow( (total_class_count/total_row),2)
        if(math.isnan(total_class_gini)):
            total_gini+=0
        else:
            total_gini += total_class_gini #adding gini to the total gini of the dataset
    return 1-total_gini


def calc_Gini(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    gini = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0] #row count of class c 
        gini_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count #probability of the class
            gini_class = pow(probability_class,2) 
            if(math.isnan(gini_class)):
                gini_class=0
        gini += gini_class
    return 1-gini

#Gini Tree

def calc_Gini_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() #unqiue values of the feature
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_gini = calc_Gini(feature_value_data, label, class_list) #calculcating gini for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_gini #calculating information of the feature value
        
    return calc_total_Gini(train_data, label, class_list) - feature_info #calculating information gain by subtracting


def find_most_Gini_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) #finding the feature names in the dataset
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  #for each feature in the dataset
        feature_info_gain = calc_Gini_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature


def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False) #dictionary of the count of unqiue feature value
    tree = {} #sub tree or node
    
    for feature_value, count in feature_value_count_dict.iteritems():
        feature_value_data = train_data[train_data[feature_name] == feature_value] #dataset with only feature_name = feature_value
        
        assigned_to_node = False #flag for tracking feature_value is pure class or not
        for c in class_list: #for each class
            class_count = feature_value_data[feature_value_data[label] == c].shape[0] #count of class c

            if class_count == count: #count of feature_value = count of class (pure class)
                tree[feature_value] = c #adding node to the tree
                train_data = train_data[train_data[feature_name] != feature_value] #removing rows with feature_value
                assigned_to_node = True
        if not assigned_to_node: #not pure class
            tree[feature_value] = "?" #should extend the node, so the branch is marked with ?
            
    return tree, train_data

def make_tree_Gini(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0: #if dataset becomes enpty after updating
        max_info_feature = find_most_Gini_feature(train_data, label, class_list) #most informative feature
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) #getting tree node and updated dataset
        next_root = None
        
        if prev_feature_value != None: #add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: #add to root of the tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()): #iterating the tree node
            if branch == "?": #if it is expandable
                feature_value_data = train_data[train_data[max_info_feature] == node] #using the updated dataset
                make_tree_Gini(next_root, node, feature_value_data, label, class_list) #recursive call with updated dataset


def id3_Gini(train_data_m, label):
    train_data = train_data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    class_list = train_data[label].unique() #getting unqiue classes of the label
    make_tree_Gini(tree, None, train_data_m, label, class_list) #start calling recursion
    return tree
#Gini Tree

def calc_total_ME(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_ME = 0
    total_class_count=0
    for c in class_list: 
        total_class_count = max(total_class_count,train_data[train_data[label] == c].shape[0]) #number of the class
    total_ME =1- total_class_count/total_row 
    return total_ME

#ME Tree 
def calc_ME(feature_value_data, label, class_list):
    targetcol=feature_value_data[label]
    elements,counts = np.unique(targetcol,return_counts = True)
    V1=1
    V1 -=counts[np.argmax(counts)]/np.sum(counts)
    return V1

def calc_ME_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() #unqiue values of the feature
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_ME = calc_ME(feature_value_data, label, class_list) #calculcating gini for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_ME #calculating information of the feature value
    
    Total=calc_total_ME(train_data, label, class_list)  
    return Total - feature_info #calculating information gain by subtracting


def find_most_ME_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) #finding the feature names in the dataset
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  #for each feature in the dataset
        feature_info_gain = calc_ME_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature

def make_tree_ME(root, prev_feature_value, train_data, label, class_list,root_feature):
    if train_data.shape[0] != 0: 
        max_info_feature = find_most_ME_feature(train_data, label, class_list) 
        if(root_feature==max_info_feature):
            return
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) 
        next_root = None
        
        if prev_feature_value != None: #add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: #add to root of the tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()): #iterating the tree node
            if branch == "?": #if it is expandable
                feature_value_data = train_data[train_data[max_info_feature] == node] #using the updated dataset
                make_tree_ME(next_root, node, feature_value_data, label, class_list,max_info_feature) #recursive call with updated dataset

def id3_ME(train_data_m, label):
    train_data = train_data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    class_list = train_data[label].unique() #getting unqiue classes of the label
    make_tree_ME(tree, None, train_data_m, label, class_list,"root") #start calling recursion
    return tree
#ME Tree 

def predict(tree, instance,level):
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    elif(level==0):
        return None
    else:
        root_node = next(iter(tree)) #getting first key/feature name of the dictionary
        feature_value = instance[root_node] #value of the feature
        if feature_value in tree[root_node]: #checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance,level-1) #goto next feature
        else:
            return None

def evaluate(tree, test_data_m, label,level):
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows(): #for each row in the dataset
        result = predict(tree, test_data_m.iloc[index],level) #predict the row
        if result == test_data_m[label].iloc[index]: #predicted value and expected value is same or not
            correct_preditct += 1 #increase correct count
        else:
            wrong_preditct += 1 #increase incorrect count
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) #calculating accuracy
    return 1-accuracy # return predicted error

def finddepth(dictA):
   if isinstance(dictA, dict): #the leaf exist
      return 1 + (max(map(finddepth, dictA.values()))
         if dictA else 0)

   return 0

column_labels = ["buying","maint","doors","persons","lug_boot","safety","label"]
label_attribute = ["unacc","acc","good","vgood"]
    
train_data_m = pd.read_csv("car\\train.csv",names=column_labels, header=None) 
test_data_m = pd.read_csv("car\\test.csv",names=column_labels, header=None) 

print("Input a level(1-6):")
level = int(input())

tree= id3_Entropy(train_data_m, column_labels[6])
treelevel=finddepth(tree)/2

print("***********************************")
print("Error Prediction of Entropy")
print("***********************************")
print("level:test data, training data")
for i in range(level+1):
    if(i==0):
        continue
    if(treelevel<i):
        break
    accuracy = evaluate(tree, test_data_m, column_labels[6],i)
    accuracy2 = evaluate(tree, train_data_m, column_labels[6],i)
    print(str(i)+":"+str(accuracy)+","+str(accuracy2))

tree= id3_Gini(train_data_m, column_labels[6])
treelevel=finddepth(tree)/2

print("***********************************")
print("Error Prediction of Gini")
print("***********************************")
print("level:test data, training data")
for i in range(level+1):
    if(i==0):
        continue
    if(treelevel<i):
        break
    accuracy = evaluate(tree, test_data_m, column_labels[6],i)
    accuracy2 = evaluate(tree, train_data_m, column_labels[6],i)
    print(str(i)+":"+str(accuracy)+","+str(accuracy2))


tree= id3_ME(train_data_m, column_labels[6])
treelevel=finddepth(tree)/2
print("***********************************")
print("Error Prediction of Majority Error")
print("***********************************")
print("level:test data, training data")
for i in range(level+1):
    if(i==0):
        continue
    if(treelevel<i):
        break
    accuracy = evaluate(tree, test_data_m, column_labels[6],i)
    accuracy2 = evaluate(tree, train_data_m, column_labels[6],i)
    print(str(i)+":"+str(accuracy)+","+str(accuracy2))