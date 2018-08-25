import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def logistic_loss(y, a):
    return -(y*math.log(a) + (1-y)*math.log(1-a))

class Perceptron:
    def __init__(self, nx, alpha = 0.01):
        self.nx = nx
        self.w = [0 for x in range(0, nx)]
        self.b = 0
        self.alpha = alpha

    def output(self, xi):
        z = sum(map(lambda z: z[0]*z[1], zip(self.w, xi))) + self.b
        return sigmoid(z)

    def predict(self, x):
        return 0 if self.output(x) < 0.5 else 1

    def deltas(self, yi, xi, a):
        loss = logistic_loss(yi, a)
        dz = a - yi
        dw = list(map(lambda x: x * dz * self.alpha, xi))
        db = dz * self.alpha
        return loss, dw, db

    def total_deltas(self, allx, ally):
        dw = [0 for x in range(0, self.nx)]
        db = 0
        cost = 0
        m = len(allx)
        for i, xi in enumerate(allx):
            a = self.output(xi)
            loss, dwi, dbi = self.deltas(ally[i], xi, a)
            dw = list(map(lambda z: z[0]+z[1], zip(dw, dwi)))
            db += dbi
            cost += loss
        dw = list(map(lambda x: x/m, dw))
        return cost/m, dw, db / m

    def adjust_params(self, dw, db):
        self.w = list(map(lambda z: z[0]-z[1], zip(self.w, dw)))
        self.b -= db
