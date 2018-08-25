
class Trainer:
    def __init__(self, perceptron, allx, ally):
        self.perceptron = perceptron
        self.allx = allx
        self.ally = ally
        self.current_step = 0
        self.history = []

    def one_step(self):
        self.current_step += 1
        cost, dw, db = self.perceptron.total_deltas(self.allx, self.ally)
        self.history.append((self.current_step, cost, dw, db))
        self.perceptron.adjust_params(dw, db)

    def train(self, nsteps):
        for i in range(0, nsteps):
            self.one_step()

    def cost_history(self):
        import matplotlib.pyplot as plt
        xx = list(map(lambda x: x[0], self.history))
        yy = list(map(lambda x: x[1], self.history))
        print(xx, yy)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xx, yy, linewidth=3)
        plt.show()

    def accuracy(self):
        correct = 0
        for i, xi in enumerate(self.allx):
            a = self.perceptron.predict(xi)
            correct += (1 if a == self.ally[i] else 0)
        return correct / len(self.allx)

