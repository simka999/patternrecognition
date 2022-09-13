import numpy as np
from prettytable import PrettyTable

# load iris dataset
# -----------------------------------------
from sklearn import datasets
iris = datasets.load_iris()

# sample test data
# -----------------------------------------
S_test = np.array([
    [7.1,3.8,6.7,2.5],
    [7.6,2.0,2.2,0.4],
    [6.2,3.1,4.1,2.4],
    [7.2,2.6,2.3,0.4],
    [6.3,2.7,4.3,0.5] ]
)
# S_test = S_test / np.array([2.3, 4, 1.5, 4]).reshape(-1, 1)
# S_test.reshape(-1, 1)

# S_test
# S_test = S_test + np.array([4, 2, 1, 0]).reshape(-1, 1)
# S_test.reshape(-1, 1)

# S_test = S_test.transpose()

# sklearn
# -----------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# split training dataset
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# training and predictions
classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
classifier.fit(X_train, Y_train)

# evaluate
# Y_prediction = classifier.predict(X_test)
# print(confusion_matrix(Y_test, Y_prediction))
# print(classification_report(Y_test, Y_prediction))
# print(accuracy_score(Y_test, Y_prediction))

# sample classification
# -----------------------------------------
Y_prediction = classifier.predict(S_test)
# print(Y_prediction)

# prettytable
# -----------------------------------------------------------
pt = PrettyTable(('sample', 'class'))
for row in list(zip(np.round(S_test, 5), Y_prediction)): pt.add_row(row)

pt.align['sample'] = 'l'
pt.align['class'] = 'c'

print(pt)

