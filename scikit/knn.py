from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClasifier(n_neighbors=5, p=2, metric='minkowski')

knn.fit(X_train, y_train)
knn.score(X_test, y_test)

