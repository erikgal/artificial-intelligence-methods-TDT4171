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

def prediction_markov(initial, transition, n, table):
    #transition
    temp = np.zeros(len(transition[0]))
    for i, value in enumerate(initial):
        P_X = np.add(temp, transition[i] * value)
        temp = P_X
    table.append(P_X.tolist())

    if n == 1:
        return table

    return prediction_markov(P_X, transition, n-1, table)

def smoothing_markov(b, e, filtered_table, transition, emission, table):
        if len(filtered_table) == 0:
            return table

        em = np.array([[emission[0][e[-1]], 0], [0, emission[1][e[-1]]]])
        b_x = transition @ em @ b

        P_X = filtered_table[-1] * b_x
        result = [round(x * 1/(sum(P_X)), 3) for x in P_X] #Normalize
        table.append(result)
        return smoothing_markov(b_x, e[:-1], filtered_table[:-1], transition ,emission, table)

def viterbi(initial, e, transition, emission, table):

    if table[0] == []: #First itteration
        temp = np.zeros(len(transition[0]))
        for i, value in enumerate(initial):
            P_X = np.add(temp, transition[i] * value)
            temp = P_X
        #emission
        P =  [emission[0][e[0]], emission[1][e[0]]]*P_X
        result = [round(x * 1/(sum(P)), 3) for x in P] #Normalize the probability values so that they add up to 1.0.

    else:
        P_X = transition[table[0][-1]]*table[1][-1]
        result = emission[e[0]]*P_X

    print(result)
    if result[0] > result[1]: #Rain is more likely
        table[0].append(0) #True
        table[1].append(result[0])
    else: #No rain is more likely
        table[0].append(1) #False
        table[1].append(result[1])

    if len(e) == 1: # Day 6
        return table

    return viterbi(result.copy(), e[1:], transition, emission, table)

def problem1b():
    print("\nProblem 1b) \n")
    F_0 = np.array([0.5, 0.5]) #Initial state
    transition = np.array([[0.8, 0.2], [0.3, 0.7]]) #Given F_{t-1} is False => P(F_t is true) = 0.3
    emission = np.array([[0.75, 0.25], [0.2, 0.8]]) #Given F_{t} is True => P(B_t is false) = 0.25
    #e = [True, True, False, True, False, True] Birds nearby => True, e[0] = e_1
    e = [0, 0, 1, 0, 1, 0]
    table = filter_markov(F_0, e, transition, emission, [])
    table.insert(0, [0.5, 0.5])
    for i, P in enumerate(table):
        print(f"P(X_{i}|e_1:{i}) = {P}")
    return table

def problem1c(filtered_table):
    print("\nProblem 1c) \n")
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    emission = np.array([[0.75, 0.25], [0.2, 0.8]])
    table = prediction_markov(filtered_table[-1], transition, 24, [])
    for i, P in enumerate(table):
        print(f"P(X_{i+7}|e_1:{i+7}) = {P}")
    x_arr = [x for x in range(31)]
    y_arr = filtered_table + table
    plt.plot(x_arr, y_arr)
    plt.ylabel('probability')
    plt.xlabel('itteration')
    plt.show()

def problem1d(filtered_table):
    print("\nProblem 1d) \n")
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    emission = np.array([[0.75, 0.25], [0.2, 0.8]])
    e = [0, 0, 1, 0, 1, 0]
    table = smoothing_markov([1,1], e, filtered_table[:-1], transition, emission, [])
    table.reverse() #Calculating from F_5 -> F_0
    for i, P in enumerate(table):
        print(f"P(X_{i}|e_1:6) = {P}")

def problem1e():
    print("\nProblem 1e) \n")

    F_0 = np.array([0.5, 0.5])
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    emission = np.array([[0.75, 0.25], [0.2, 0.8]])
    e = [0, 0, 1, 0, 1, 0]
    table = viterbi(F_0, e, transition, emission, [[],[]]) #For 2 possible paths

    boolean_table = []
    for boolean in table[0]:
        if boolean == 0:
            boolean_table.append(True)
        else:
            boolean_table.append(False)

    print("The most likely sequence is:", boolean_table)

if __name__ == '__main__':
    filtered_table = problem1b()
    problem1c(filtered_table)
    problem1d(filtered_table)
    problem1e()
