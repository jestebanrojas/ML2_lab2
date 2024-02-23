import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iters=1000):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        # Random centroid choose
        self.centroids = X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]

        for _ in range(self.max_iters):
            # Calcular las distancias entre los puntos y los centroides
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

            # Asignar cada punto al clúster más cercano
            labels = np.argmin(distances, axis=1)

            # Actualizar los centroides
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])

            # Si los centroides no cambian mucho, salir del bucle
            if np.linalg.norm(new_centroids - self.centroids) < 1e-4:
                break

            self.centroids = new_centroids

        self.labels_ = labels
        return self