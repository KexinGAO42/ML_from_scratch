import numpy as np


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, train_data):
        # Step 1: initialize k centroids
        selected_index = np.random.choice(train_data.shape[0], self.k, replace=False)
        self.centroids = train_data[selected_index, :]

        # Step 2: iterate the process to update centroids for max_iter times
        for i in range(self.max_iter):
            self.classification = {}

            for j in range(self.k):
                self.classification[j] = []

            for data in train_data:
                distances = [np.linalg.norm(data - centroid) for centroid in self.centroids]  # calculate the distance
                classification = distances.index(min(distances))  # get the closest centroid
                self.classification[classification].append(data)  # append the data point to this centroid

            prev_centroids = self.centroids

            for classification in self.classification.keys():  # update centroids
                self.centroids[classification] = np.average(self.classification[classification], axis=0)

            # Step 3: early stop when the moving distance is smaller than tol
            optimized = True

            for i in range(self.k):
                crt_centroid = self.centroids[i]
                prev_centroid = prev_centroids[i]
                if np.sum((crt_centroid - prev_centroid) / prev_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, test_data):
        y = []
        for data in test_data:
            distances = [np.linalg.norm(data - centroid) for centroid in self.centroids]
            y.append(distances.index(min(distances)))
        return np.array(y)


train = np.random.rand(700, 2)
test = np.random.rand(300, 2)
model = K_Means(k=5, tol=0.01, max_iter=5)
model.fit(train)
result = model.predict(test)
print(result)

