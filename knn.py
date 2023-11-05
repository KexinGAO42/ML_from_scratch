import numpy as np
from collections import Counter


def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 -x2) ** 2))


def manhattan_dist(x1, x2):
    return np.sum(abs(x1 - x2))


class k_NN():
    def __init__(self, k=2, dist="euclidean"):
        self.k = k
        self.dist = dist

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y = []
        for x in X:
            y.append(self._cal_dist(x))
        return np.array(y)

    def _cal_dist(self, x):
        if self.dist == "euclidean":
            distances = [euclidean_dist(x, x_train) for x_train in self.X_train]
        if self.dist == "manhattan":
            distances = [manhattan_dist(x, x_train) for x_train in self.X_train]
        top_k = np.argsort(distances)[:self.k]
        knn_y = [self.y_train[i] for i in top_k]
        count_sort = Counter(knn_y).most_common(1)
        return count_sort[0][0]


model = k_NN(k=5, dist="manhattan")
train_X = np.random.rand(700, 2)
train_y = np.random.randint(10, size=700)
test_X = np.random.rand(300, 2)
model.fit(train_X, train_y)
result = model.predict(test_X)
print(result)
