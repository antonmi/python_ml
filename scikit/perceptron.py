from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
ppn.score(X_test_std, y_test)

y_pred = ppn.predict(X_test_std)
misclassified = (y_test != y_pred).sum()
accuracy = (y_test.size - misclassified) / y_test.size

import sklearn.metrics
sklearn.metrics.accuracy_score(y_test, y_pred)