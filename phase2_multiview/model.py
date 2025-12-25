import numpy as np

class SizeRegressor:
    def fit(self, X, y):
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        self.W = np.linalg.pinv(Xb) @ y

    def predict(self, X):
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        return Xb @ self.W
