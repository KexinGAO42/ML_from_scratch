import numpy as np


class Gaussian_NB():

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, prior for each class
        # initialization
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for i, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[i, :] = X_c.mean(axis=0)
            self._var[i, :] = X_c.var(axis=0)
            self._priors[i] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._get_prediction(x) for x in X]
        return np.array((y_pred))

    def _get_prediction(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._gaussian_pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # select the class with max posterior as prediction
        return self._classes[np.argmax(posteriors)]

    def _gaussian_pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        print(y_true, y_pred)
        return accuracy

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    model = Gaussian_NB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))

