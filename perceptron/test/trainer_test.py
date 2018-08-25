import test_helper
import numpy as np
import pandas as pd
import perceptron, trainer

def allx():
    return np.array([[2, 1], [3, 2], [1, 2], [2, 3]])

def ally():
    return np.array([1, 1, 0, 0])

def almost_eq(var, target, delta = 0.001):
    return abs(var - target) < delta

def test_one_step():
    w, b = np.array([.0, .0]), .0
    w, b, cost = trainer.one_step(allx(), ally(), w, b)
    assert almost_eq(w[0], 0.0025)
    assert almost_eq(w[1], -0.0025)
    assert almost_eq(b, 0.0)
    assert almost_eq(cost, 0.6931)

def test_train():
    w, b = np.array([.0, .0]), .0
    w, b, costs = trainer.train(allx(), ally(), w, b, 1000)
    assert almost_eq(costs[-1], 0.2)

    predictions = perceptron.predict(np.array([[3,1], [1,3]]), w, b)
    assert predictions[0] == 1
    assert predictions[1] == 0

def test_plot_costs():
    w, b = np.array([.0, .0]), .0
    w, b, costs = trainer.train(allx(), ally(), w, b, 1000)
    # trainer.plot_costs(costs)

# iris test set
def setosa_versicolor():
    df = pd.read_csv('iris.csv')
    df = df.loc[df['species'].isin(['setosa', 'versicolor'])]
    df.loc[df['species'] == 'setosa', 'species'] = 0
    df.loc[df['species'] == 'versicolor', 'species'] = 1
    return df

def iris_allx():
    df = setosa_versicolor()
    return np.array(df.iloc[:, 0:4].values.tolist())

def iris_ally():
    df = setosa_versicolor()
    return np.array(df.iloc[:, 4].values.tolist())

def test_iris_set():
    w, b = np.array([.0, .0, .0, .0]), .0
    w, b, costs = trainer.train(iris_allx(), iris_ally(), w, b, 1000)
    trainer.plot_costs(costs)