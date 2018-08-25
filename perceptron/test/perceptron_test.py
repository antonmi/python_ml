import test_helper
import perceptron
import numpy as np

def test_sigmoid():
    assert perceptron.sigmoid(10) > 0.999

def almost_eq(var, target, delta = 0.001):
    return abs(var - target) < delta

def allx():
    return np.array([[2, 1], [3, 2], [1, 2], [2, 3]])

def ally():
    return np.array([1, 1, 0, 0])

def test_output():
    w = np.array([1,0])
    b = 2
    output = perceptron.output(allx(), w, b)
    assert output.shape == (4,)
    assert np.all(output < 1)

def test_predict():
    xx = np.array([[3,1], [1,3]])
    w = np.array([3,-3])
    b = 0
    predictions = perceptron.predict(xx, w, b)
    assert predictions[0] == 1
    assert predictions[1] == 0

def test_grads():
    alla = np.array([0.9, 0.5, 0.5, 0.8])
    dw, db = perceptron.grads(ally(), allx(), alla)
    assert almost_eq(dw[0], 0.001)
    assert almost_eq(dw[1], 0.00575)
    assert almost_eq(db, 0.00175)

def test_apply_grads():
    w, dw = np.array([1.0, 1.0]), np.array([0.1, 0.1])
    b, db = 1, 0.1
    w, b = perceptron.apply_grads(w, b, dw, db)
    assert w[0] == 0.9
    assert w[1] == 0.9
    assert b == 0.9

def test_cost():
    alla = np.array([0.9, 0.5, 0.5, 0.8])
    cost = perceptron.cost(ally(), alla)
    assert almost_eq(cost, 0.775)
