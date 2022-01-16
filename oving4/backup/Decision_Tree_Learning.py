import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator, uuid
from graphviz import Digraph
import random

titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')

def get_distinct_val(data):
    dict = {}
    for col in data.columns:
        if len(data[col].unique()) < 30:
            dict[col] = data[col].unique()
    return dict

distinct_val = get_distinct_val(titanic_train)

def get_cont_att(data):
    lst = []
    for col in data.columns:
        if len(data[col].unique()) > 30:
            lst.append(col)
    return lst

cont_val = get_cont_att(titanic_train)
#Define B(q) as the entropy of a Boolean random variable that is true with probability q
def B(q):
    if q == 1:
        return 1
    elif q == 0:
        return 0

    return -(q*np.log2(q) + (1-q)*np.log2(1-q))

#If a training set contains p positive examples and n negative examples,
#then the entropy of the goal attribute on the whole set is
def Remainder(T, A, p, n, examples):
    #An attribute A with d distinct values divides the training set E into subsets E1,...,Ed
    distinct_val = examples[A].unique()
    splits = []
    for val in distinct_val:
        splits.append(examples[examples[A] == val]) #DataFrame with only distinct_val

    #We sum over all possible values A kan take like Sex: F, M, Uknown
    #pk are for all values where e.g. Females survive
    sum = 0
    for split in splits:
        try:
            pk = split[T].value_counts()[0]
        except:
            pk = 0
        try:
            nk = split[T].value_counts()[1]
        except:
            nk = 0

        sum += ((pk + nk)/(p + n)) * B(pk/(pk + nk))
    return sum

#Gain(A) = B(p/(p+n)) − Remainder(A) s704
def information_gain(T, A, examples):
    p = examples[T].value_counts()[0]
    n = examples[T].value_counts()[1]

    return B(p/(p+n)) - Remainder(T, A, p, n, examples)

def cont_Remainder(T, A, p, n, midpoint, examples):
    splits =  []
    splits.append(examples[examples[A] < midpoint]) #low-end split
    splits.append(examples[examples[A] > midpoint]) #high-end split

    sum = 0
    for split in splits:
        try:
            pk = split[T].value_counts()[0]
        except:
            pk = 0
        try:
            nk = split[T].value_counts()[1]
        except:
            nk = 0

        sum += ((pk + nk)/(p + n)) * B(pk/(pk + nk))
    return sum

def cont_information_gain(T, A, examples):
    p = examples[T].value_counts()[0]
    n = examples[T].value_counts()[1]

    #Find all split points to evaluate
    unique_ages = examples.sort_values(by=[A])[A].unique()
    splits = []
    for i in range(len(unique_ages)-1):
        if not np.isnan(unique_ages[i]) and not np.isnan(unique_ages[i+1]):
            midpoint = (unique_ages[i] + unique_ages[i+1])/2
            splits.append(midpoint)

    remainders = {}
    for split in splits:
        remainder = cont_Remainder(T, A, p, n, split, examples)
        remainders[split] = remainder

    min_remainder = min(remainders.items(), key=operator.itemgetter(1)) # [age, remainder]
    return [min_remainder[0], B(p/(p+n)) - min_remainder[1]] # [split age, gain]

def plurality_value(examples, target, graph):
    value = examples[target].value_counts().idxmax()

    if value != 0 and value != 1:
        value = random.randint(0,1)

    if value == 0:
        id = str(uuid.uuid1())
        graph.node(id, label = 'Died')
        return [id, 'Died']

    id = str(uuid.uuid1())
    graph.node(id, label = 'Survived')
    return [id, 'Survived']

def DTL(examples, attributes, parent_examples, target, graph):
    if examples.empty: #if examples is empty then return PLURALITY-VALUE(parent examples)
        return plurality_value(parent_examples, target, graph)

    elif len(examples[target].unique()) == 1: #if all examples have the same classification then return the classification
        value = (examples[target].unique()[0])
        if value == 0:
            id = str(uuid.uuid1())
            graph.node(id, label = 'Died')
            return [id, 'Died']
        id = str(uuid.uuid1())
        graph.node(id, label = 'Survived')
        return [id, 'Survived']
    elif attributes == []: #if attributes is empty
        return plurality_value(parent_examples, target, graph)

    gain = {}
    dict = {}
    cont = [] #Saves all continuous attributes
    for col in attributes:
        if col in cont_val: #is continuous
            results = cont_information_gain('Survived', col, examples)
            midpoint = results[0]
            gain[col] = results[1]
            cont.append(col)
        else: #is discrete
            gain[col] = information_gain('Survived', col, examples)
    max_gain = max(gain.items(), key=operator.itemgetter(1))[0]
    id = str(uuid.uuid1()) #Unique identifier

    graph.node(id, label = max_gain)
    dict_list = {}
    if max_gain in cont: #If splitting on continuous attribute
        #Greater
        exs = examples[examples[max_gain] < midpoint]
        new_attributes = attributes.copy()
        new_attributes.remove(max_gain)
        subtree = DTL(exs, new_attributes, examples, target, graph)
        graph.edge(id, subtree[0], label = f" < {midpoint}")
        dict_list[f"< {midpoint}"] = subtree[1]
        #Lesser
        exs = examples[examples[max_gain] > midpoint]
        new_attributes = attributes.copy()
        new_attributes.remove(max_gain)
        subtree = DTL(exs, new_attributes, examples, target, graph)
        graph.edge(id, subtree[0], label = f" > {midpoint}")
        dict_list[f"> {midpoint}"] = subtree[1]

    else: #If splitting on categorical attribute
        for vk in distinct_val[max_gain]:
            exs = examples[examples[max_gain] == vk] #{e : e ∈ examples and e.A = vk}
            new_attributes = attributes.copy()
            new_attributes.remove(max_gain)
            subtree = DTL(exs, new_attributes, examples, target, graph)
            graph.edge(id, subtree[0], label = str(vk))
            dict_list[vk] = subtree[1]

    dict[max_gain] = dict_list
    return [id, dict]

def itterate_row(dict, row):
    for key in dict:
        if dict == 'Died' or dict == 'Survived':
            return dict
        direction = row.loc[key]

        if key in cont_val:
            """
            if np.isnan(direction): #TODO
                return 'Died'
            """
            expression = eval(str(direction) + list(dict[key].keys())[0]) #i.e. 35.0 < 8.5
            if expression:
                return itterate_row(dict[key][list(dict[key].keys())[0]], row)
            else:
                return itterate_row(dict[key][list(dict[key].keys())[1]], row)

    return itterate_row(dict[key][direction], row) #Recursiv call

def DTL_test(dtl, test_set):
    Survival = {0: 'Died', 1: 'Survived'}
    results = [0, 0] #[true, false]
    for (idx, row) in test_set.iterrows():
        result = itterate_row(dtl, row) #What the DTL thinks is going to happen
        if result == Survival[row.loc['Survived']]: #Checking if it predicted right
            results[0] += 1
        else:
            results[1] += 1
    return results

if __name__ == '__main__':
    # a)
    graph = Digraph('Titanic', filename='titanic_graph.gv')
    dict = {}
    attributes = ['Pclass','SibSp', 'Sex', 'Embarked', 'Parch'] #['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
    result = DTL(titanic_train, attributes, None, 'Survived', graph)
    decision_dict = result[1]
    graph.view()

    results = DTL_test(decision_dict, titanic_test)
    print("The regulare DTL predicted \nCorrect:", results[0], "\nWrong:", results[1],
     "\nAccuracy:",  round(results[0]/(results[0]+results[1])*100, 2), "%\n \n \n")

    # b)
    graph = Digraph('Titanic', filename='titanic_cont_graph.gv')
    dict = {}
    attributes = ['Pclass','SibSp', 'Parch', 'Sex', 'Embarked', 'Fare'] #['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Fare']
    result = DTL(titanic_train, attributes, None, 'Survived', graph)
    decision_dict = result[1]
    graph.view()

    results = DTL_test(decision_dict, titanic_test)
    print("The continuous DTL predicted \nCorrect:", results[0], "\nWrong:", results[1],
     "\nAccuracy:",  round(results[0]/(results[0]+results[1])*100, 2), "%")
