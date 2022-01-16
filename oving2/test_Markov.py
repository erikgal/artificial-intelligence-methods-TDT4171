import numpy as np
import matplotlib.pyplot as plt

#P(X_{t+k+1}|e_1:t) = Σ_{x_{t+k}} P(X_{t+k+1}|x_{t+k} )P(x_{t+k} |e_{1:t})
def filter_markov(initial, e, transition, emission, table):
    #transition
    temp = np.zeros(len(transition[0]))
    for i, value in enumerate(initial):
        P_X = np.add(temp, transition[i] * value)
        temp = P_X

    #emission
    P =  [emission[0][e[0]], emission[1][e[0]]]*P_X
    result = [round(x * 1/(sum(P)), 3) for x in P] #Normalize the probability values so that they add up to 1.0.
    table.append(result)

    if len(e) == 1: # Day 6
        return table

    return filter_markov(result.copy(), e[1:], transition, emission, table)

def viterbi(initial, e, transition, emission, table):

    if table == []: #First itteration
        temp = np.zeros(len(transition[0]))
        for i, value in enumerate(initial):
            P_X = np.add(temp, transition[i] * value)
            temp = P_X
        #emission
        P =  [emission[0][e[0]], emission[1][e[0]]]*P_X
        result = [round(x * 1/(sum(P)), 4) for x in P] #Normalize the probability values so that they add up to 1.0.

        print("r:", result, "\n")
        table.append(result)
        if len(e) == 1: # Day 6
            return table

        return viterbi(result.copy(), e[1:], transition, emission, table)

    else:
        result = []
        print(e[0])
        for i in range(len(transition)):
            temp = []
            for j, prob in enumerate(table[-1]):
                print(transition[j][i], "*",emission[i][e[0]], "*", prob, "=", transition[j][i] * emission[i][e[0]] * prob)
                temp.append(transition[j][i] * emission[i][e[0]] * prob)
            result.append(round(max(temp), 4))
        """
        for i in range(len(transition)):
            print(transition[i], "*", emission[i], "*", table[-1], "=", transition[i]* emission[i] * table[-1])
            delta = transition[i]* [emission[i]] * table[-1]
            result.append(round(max(delta), 4))
        """
    print("r:", result, "\n")
    table.append(result)
    if len(e) == 1: # Day 6
        return table

    return viterbi(result.copy(), e[1:], transition, emission, table)

def problem1b():
    print("\nProblem 1b) \n")
    F_0 = np.array([0.5, 0.5]) #Initial state
    transition = np.array([[0.7, 0.3], [0.3, 0.7]]) #Given F_{t-1} is False => P(F_t is true) = 0.3
    emission = np.array([[0.9, 0.2], [0.2, 0.1]]) #Given F_{t} is True => P(B_t is false) = 0.25
    #e = [True, True, False, True, False, True] Birds nearby => True, e[0] = e_1
    e = [0, 0]
    table = filter_markov(F_0, e, transition, emission, [])
    table.insert(0, [0.5, 0.5])
    for i, P in enumerate(table):
        print(f"P(X_{i}|e_1:{i}) = {P}")
    return table

def problem1d(filtered_table):
    print("\nProblem 1d) \n")
    transition = np.array([[0.7, 0.3], [0.3, 0.7]])
    emission = np.array([[0.9, 0.2], [0.2, 0.1]])
    e = [0]
    table = smoothing_markov([1,1], e, filtered_table[:-1], transition, emission, [])
    table.reverse() #Calculating from F_5 -> F_0
    for i, P in enumerate(table):
        print(f"P(X_{i}|e_1:2) = {P}")

def problem1e():
    print("\nProblem 1e) \n")

    F_0 = np.array([0.5, 0.5])
    transition = np.array([[0.7, 0.3], [0.3, 0.7]])
    emission = np.array([[0.9, 0.1], [0.2, 0.8]])
    e = [0, 0, 1, 0, 0]
    table = viterbi(F_0, e, transition, emission, []) #For 2 possible paths
    print(table)
    boolean_table = []
    for boolean in table[0]:
        if boolean == 0:
            boolean_table.append(True)
        else:
            boolean_table.append(False)

    print("The most likely sequence is:", boolean_table)


if __name__ == '__main__':
    #filtered_table = problem1b()
    problem1e()
