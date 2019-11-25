"""
9. Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Print
both correct and wrong predictions. Java/Python ML library classes can be used for this
problem.
"""

import csv
import numpy as np
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class_dict = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
with open('ds5.csv') as csvFile:
    dataset = [line for line in csv.reader(csvFile)]
    dataset = dataset[1:]
    X = []
    y = []
    for line in dataset:
        X.append(line[:-1])
        y.append(class_dict[line[-1]])

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y)

classifier = KNeighborsClassifier(n_neighbors=5)

y_pred = classifier.fit(X_train, y_train).predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

print(class_dict)
print("X", "y_actual", "y_predicted", "is_correct")
for _x, _ya, _yp in zip(X_test, y_test, y_pred):
    print(_x, _ya, _yp, _ya == _yp)

"""
Output:

Accuracy:  0.9736842105263158
{'setosa': 0, 'versicolor': 1, 'virginica': 2}
X y_actual y_predicted is_correct
[5.8 2.8 5.1 2.4] 2 2 True
[6.  2.2 4.  1. ] 1 1 True
[5.5 4.2 1.4 0.2] 0 0 True
[7.3 2.9 6.3 1.8] 2 2 True
[5.  3.4 1.5 0.2] 0 0 True
[6.3 3.3 6.  2.5] 2 2 True
[5.  3.5 1.3 0.3] 0 0 True
[6.7 3.1 4.7 1.5] 1 1 True
[6.8 2.8 4.8 1.4] 1 1 True
[6.1 2.8 4.  1.3] 1 1 True
[6.1 2.6 5.6 1.4] 2 2 True
[6.4 3.2 4.5 1.5] 1 1 True
[6.1 2.8 4.7 1.2] 1 1 True
[6.5 2.8 4.6 1.5] 1 1 True
[6.1 2.9 4.7 1.4] 1 1 True
[4.9 3.1 1.5 0.1] 0 0 True
[6.  2.9 4.5 1.5] 1 1 True
[5.5 2.6 4.4 1.2] 1 1 True
[4.8 3.  1.4 0.3] 0 0 True
[5.4 3.9 1.3 0.4] 0 0 True
[5.6 2.8 4.9 2. ] 2 2 True
[5.6 3.  4.5 1.5] 1 1 True
[4.8 3.4 1.9 0.2] 0 0 True
[4.4 2.9 1.4 0.2] 0 0 True
[6.2 2.8 4.8 1.8] 2 2 True
[4.6 3.6 1.  0.2] 0 0 True
[5.1 3.8 1.9 0.4] 0 0 True
[6.2 2.9 4.3 1.3] 1 1 True
[5.  2.3 3.3 1. ] 1 1 True
[5.  3.4 1.6 0.4] 0 0 True
[6.4 3.1 5.5 1.8] 2 2 True
[5.4 3.  4.5 1.5] 1 1 True
[5.2 3.5 1.5 0.2] 0 0 True
[6.1 3.  4.9 1.8] 2 2 True
[6.4 2.8 5.6 2.2] 2 2 True
[5.2 2.7 3.9 1.4] 1 1 True
[5.7 3.8 1.7 0.3] 0 0 True
[6.  2.7 5.1 1.6] 1 2 False
"""
