import unittest
from perceptron import Perceptron
from trainer import Trainer
import pdb

class TrainerTest(unittest.TestCase):
    def allx(self):
        return [[2, 1], [3, 2], [1, 2], [2, 3]]

    def ally(self):
        return [1,1,0,0]

    def test_one_step(self):
        per = Perceptron(2)
        w0 = per.w
        b0 = per.b
        trainer = Trainer(per, self.allx(), self.ally())
        trainer.one_step()
        self.assertNotEqual(w0, per.w)

    def test_train(self):
        per = Perceptron(2)
        trainer = Trainer(per, self.allx(), self.ally())
        trainer.train(5000)
        self.assertEqual(per.predict([3,1]), 1)
        self.assertEqual(per.predict([1,3]), 0)

import pandas as pd
class IrisTest(unittest.TestCase):

    def setosa_versicolor(self):
        df = pd.read_csv('iris.csv')
        df = df.loc[df['species'].isin(['setosa', 'versicolor'])]
        df.loc[df['species'] == 'setosa', 'species'] = 0
        df.loc[df['species'] == 'versicolor', 'species'] = 1
        return df

    def allx(self):
        df = self.setosa_versicolor()
        return df.iloc[:, 0:4].values.tolist()

    def ally(self):
        df = self.setosa_versicolor()
        return df.iloc[:, 4].values.tolist()

    def test_iris_set(self):
        per = Perceptron(5)
        trainer = Trainer(per, self.allx(), self.ally())
        trainer.train(5000)
        self.assertEqual(per.predict([6.9,3.1,5.1,2.3]), 1)
        print(trainer.accuracy())

unittest.main()