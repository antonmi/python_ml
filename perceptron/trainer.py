import perceptron

def one_step(allx, ally, w, b):
    alla = perceptron.output(allx, w, b)
    dw, db = perceptron.grads(ally, allx, alla)
    cost = perceptron.cost(ally, alla)
    w, b = perceptron.apply_grads(w, b, dw, db)
    return w, b, cost

def train(allx, ally, w, b, steps = 100):
    costs = []
    for i in range(0, steps):
        w, b, cost = one_step(allx, ally, w, b)
        costs.append(cost)
    return w, b, costs

def plot_costs(costs):
    import matplotlib.pyplot as plt
    xx = list(range(0, len(costs)))
    yy = costs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, yy, linewidth=3)
    plt.show()
