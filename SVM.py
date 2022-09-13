import numpy as np 
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import zero_one_score


X = [[14.5], [5.5], [7.5], [10.5], [15.5], [13.5], [9.5], [8.5], [10.5], [16.5], [14.5], [11.5], [13.5], [11.5], [16.5], [13.5], [18.5], [10.5]]

y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

clf = svm.SVC(kernel="linear")
clf.fit(X, y)
w = clf.coef_[0]
supportvects = clf.support_vectors_


margin = 2 / np.sqrt(np.sum(w**2))

y_pred = svm.predict(test_samples)
accuracy = zero_one_score(y_test, y_pred)
error_rate = 1 - accuracy
# print(clf.predict([[2.5, 0.5]]))
# print(margin)
# print(supportvects)
print(error_rate)
