import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_loss(y, a):
    return -(y*np.log(a) + (1-y)*np.log(1-a))

def output(allx, w, b):
    z = np.dot(allx, w.transpose()) + b
    return sigmoid(z)

def predict(x, w, b):
    return np.where(output(x, w, b) > 0.5, 1,0)

def grads(ally, allx, alla, alpha = 0.01):
    m = alla.shape[0]
    dz = alla - ally
    dw = np.sum(np.multiply(allx, dz.reshape(m, 1)), axis = 0) * alpha / m
    db = np.sum(dz) * alpha / m
    return dw, db

def cost(ally, alla):
    m = alla.shape[0]
    return np.sum(logistic_loss(ally, alla)) / m

def apply_grads(w, b, dw, db):
    w += -dw
    b += -db
    return w, b