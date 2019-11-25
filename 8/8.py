"""
8. Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set for
clustering using k-Means algorithm. Compare the results of these two algorithms and comment
on the quality of clustering. You can add Java/Python ML library classes/API in the program.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

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


def rename_clusters(s):
    l2 = []
    for i in s:
        if i not in l2:
            l2.append(i)

    for i in range(len(s)):
        pos = l2.index(s[i])
        s[i] = pos
    return s


# EM part
gmm = GaussianMixture(n_components=3)
y_kmeans = gmm.fit(X).predict(X)
y_kmeans = rename_clusters(y_kmeans)

print("Accuracy EM : ", metrics.accuracy_score(y, y_kmeans))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
plt.show()

# K-means part
kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit(X).predict(X)
y_kmeans = rename_clusters(y_kmeans)

print("Accuracy KM : ", metrics.accuracy_score(y, y_kmeans))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
plt.show()

"""
Output:

Accuracy EM :  0.9666666666666667
Accuracy KM :  0.8933333333333333
"""
