import numpy as np



class Perceptron:
    def __init__(self, lr=1e-2, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.act = lambda x: np.where(x >= 0, 1, 0)

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0

        y = np.array([1 if x > 0 else 0 for x in y])

        for _ in range(self.n_iters):
            for i, x in enumerate(X):
                update = self.lr * (y[i] - self.predict(x))
                self.weights += update * x
                self.bias += update

    def predict(self, X):
        return self.act(np.dot(X, self.weights) + self.bias)


