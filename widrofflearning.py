import numpy as np
from prettytable import PrettyTable

# Write code to apply 2 epochs of the Sequential Widrow-Hoff Learning Algorithm to the training data shown in the table. Use initial parameters values of aT=(w0,wT)=(1.0,0.0,0.0), use a margin vector bT=(1.0,0.5,0.5,0.5,0.5,1.0), and use a learning rate of 0.1.

# configuration variables
# -----------------------------------------------------------
# initial values
a = [-0.5, 0, 0.5]

# margin vector
b = [1, 1, 1, 1, 1, 1]

# learning rate
n = 1

# iterations
iterations = 12

# dataset
# -----------------------------------------------------------
X = [[0, 1], [1, 0], [0.5, 1.5], [1, 1], [-0.5,0]]
Y = [[1, 0, 1], [1, 1, 0], [1, 0.5, 1.5], [1, 1, 1], [1, -0.5,0]]

# sequential widrow-hoff learning algorithm
# -----------------------------------------------------------
result = []
for o in range(int(iterations / len(Y))):
    for i in range(len(Y)):
        a_prev = a
        y = Y[i]

        # calculate ay
        ay = np.dot(a, y)

        # calculate update part
        update = np.zeros(len(y))
        for j in range(len(y)): update[j] = n * (b[i] - ay) * y[j]

        # add update part to a
        a = np.add(a, update)

        # append result
        result.append((str(i + 1 + (len(Y) * o)), np.round(a_prev, 4), np.round(y, 4), np.round(ay, 4), np.round(a, 4)))

# prettytable
# -----------------------------------------------------------
pt = PrettyTable(('iteration', 'a', 'y', 'ay', 'a_new'))
for row in result: pt.add_row(row)

pt.align['iteration'] = 'c'
pt.align['a'] = 'l'
pt.align['y'] = 'l'
pt.align['ay'] = 'l'
pt.align['a_new'] = 'l'

print(pt)