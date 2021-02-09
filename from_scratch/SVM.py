import numpy as np

class SVM:
    def __init__(self, lr=1e-3, lam=1e-2, n_iters=1000):
        self.lr = lr
        self.lam = lam
        self.n_iters = n_iters

        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)
        self.w = np.random.randn(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for i, x in enumerate(X):
                self.w -= self.lr * 2 * self.lam * self.w
                if y[i] * self._score(x) < 1:
                    self.w += self.lr * x * y[i]
                    self.b -= self.lr * y[i]

    def predict(self, X):
        return np.sign(self._score(X))

    def _score(self, x):
        return np.dot(x, self.w) - self.b
