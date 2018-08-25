from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

svm.score(X_test_std, y_test)

y_pred = svm.predict(X_test_std)
misclassified = (y_test != y_pred).sum()
accuracy = (y_test.size - misclassified) / y_test.size

#Nonliner data

X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:,1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.show()

svm.fit(X_xor, y_xor)
svm.score(X_xor, y_xor)

svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)

from plot_helper import plot_decision_regions
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()
