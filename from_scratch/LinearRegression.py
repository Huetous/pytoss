import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def noop(x):
    return x


class BaseLogisticRegression:
    def __init__(self, lr=1e-3, n_iters=1000, act=noop):
        self.lr = lr
        self.n_iters = n_iters

        self.weights = None
        self.bias = None

        self.act = act

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            pred = self.act((np.dot(X, self.weights) + self.bias))

            dw = (2 / n_samples) * np.dot(X.T, (pred - y))
            db = (2 / n_samples) * np.sum(pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return self.act(np.dot(X, self.weights) + self.bias)


class LinearRegression(BaseLogisticRegression):
    def __init__(self, lr=1e-3, n_iters=1000):
        super().__init__(lr, n_iters, act=noop)


class LogisticRegression(BaseLogisticRegression):
    def __init__(self, lr=1e-3, n_iters=1000):
        super().__init__(lr, n_iters, act=sigmoid)

    def predict(self, X):
        return np.round(super().predict(X))
