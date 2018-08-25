import unittest
from perceptron import *

class PerceptronTest(unittest.TestCase):
    def almost_eq(self, var, target, delta = 0.001):
        return abs(var - target) < delta

    def test_output(self):
        p = Perceptron(2)
        p.w = [0,1]
        p.b = 2
        output = p.output([1,1])
        self.assertTrue(1 - output < 0.05)

    def test_deltas(self):
        p = Perceptron(2, 0.1)
        loss, dw, db = p.deltas(0, [1,1], 0.9)
        self.assertTrue(self.almost_eq(loss, 2.3025))
        self.assertTrue(self.almost_eq(dw[0], 0.09))
        self.assertTrue(self.almost_eq(dw[1], 0.09))
        self.assertTrue(self.almost_eq(db, 0.09))

    def test_total_deltas(self):
        p = Perceptron(2, 0.1)
        p.w = [0.1, 0.1]
        allx = [[1,2], [1,0], [0, 1]]
        ally = [1, 0, 1]
        cost, dw, db = p.total_deltas(allx, ally)
        self.assertTrue(self.almost_eq(cost, 0.648))
        self.assertTrue(self.almost_eq(dw[0], 0.0033))
        self.assertTrue(self.almost_eq(dw[1], -0.044))
        self.assertTrue(self.almost_eq(db, -0.0125))

unittest.main()