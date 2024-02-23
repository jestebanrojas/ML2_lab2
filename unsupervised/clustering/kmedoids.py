import numpy as np

class KMedoids:
    def __init__(self, n_clusters=2, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.cluster_centers_ = self._initialize_medoids()
        self.labels_ = np.zeros(self.n_samples)

        for _ in range(self.max_iter):
            old_cluster_centers = np.copy(self.cluster_centers_)
            self.labels_ = self._assign_clusters()
            self._update_medoids()

            if np.all(old_cluster_centers == self.cluster_centers_):
                break

        return self

    def _initialize_medoids(self):
        indices = np.random.choice(self.n_samples, self.n_clusters, replace=False)
        return self.X[indices]

    def _assign_clusters(self):
        distances = np.zeros((self.n_samples, self.n_clusters))

        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(self.X - self.cluster_centers_[i], axis=1)

        return np.argmin(distances, axis=1)

    def _update_medoids(self):
        for i in range(self.n_clusters):
            cluster_points = self.X[self.labels_ == i]
            medoid_indices = np.arange(len(cluster_points))
            medoid_distances = np.zeros(len(cluster_points))

            for j, point in enumerate(cluster_points):
                medoid_distances[j] = np.sum(np.linalg.norm(cluster_points - point, axis=1))

            new_medoid_index = medoid_indices[np.argmin(medoid_distances)]
            self.cluster_centers_[i] = cluster_points[new_medoid_index]

    def predict(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))

        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.cluster_centers_[i], axis=1)

        return np.argmin(distances, axis=1)