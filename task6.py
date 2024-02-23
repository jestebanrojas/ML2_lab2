import numpy as np
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None
# Anisotropically distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

"""
--------------------------------------------------------------------------------------
a. Plot the different datasets in separate figures. What can you say about them?
--------------------------------------------------------------------------------------
"""

# Plotting datasets
datasets = [noisy_circles, noisy_moons, blobs, aniso, varied]
dataset_names = ['Noisy Circles', 'Noisy Moons', 'Blobs', 'Anisotropic', 'Varied Variances']

fig, axs = plt.subplots(5, 1, figsize=(8, 30))

for i, dataset in enumerate(datasets):
    X, y = dataset

    axs[i].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, edgecolors='k')
    axs[i].set_title(f'Dataset: {dataset_names[i]}')
    axs[i].set_xlabel('Feature 1')
    axs[i].set_ylabel('Feature 2')



# Add space between subplots
plt.subplots_adjust(hspace=1)
plt.show()


"""
--------------------------------------------------------------------------------------
b. Apply k-means, k-medoids, DBSCAN and Spectral Clustering from Scikit-Learn over each
dataset and compare the results of each algorithm with respect to each dataset.
--------------------------------------------------------------------------------------
"""




# Define clustering algorithms
algorithms = {
    'K-Means': KMeans(n_clusters=3),  
    'K-Medoids': KMedoids(n_clusters=3),  
    'DBSCAN': DBSCAN(eps=0.2, min_samples=6),
    'Spectral Clustering': SpectralClustering(n_clusters=3, affinity='rbf')  
}

# Define datasets
datasets = {
    'Noisy Circles': noisy_circles,
    'Noisy Moons': noisy_moons,
    'Blobs': blobs,
    'Anisotropic': aniso,
    'Varied Variances': varied
}

fig, axs = plt.subplots(4, 5, figsize=(30, 30))

# Evaluate and compare clustering results
i=0

for dataset_name, (X, y) in datasets.items():
    print(f"\nDataset: {dataset_name}")
    j=0
    for algo_name, algo in algorithms.items():
        # Special case for DBSCAN where labels can be -1 (outliers)
        labels = algo.fit_predict(X)

        # Skip silhouette score calculation for DBSCAN if only one label is assigned
        if algo_name == 'DBSCAN' and len(np.unique(labels)) == 1:
            print(f"{algo_name}: Only one cluster detected (possibly all outliers)")
            j=j+1
            continue

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, labels)
        print(f"{algo_name}: Silhouette Score = {silhouette_avg}")

        # Plot the clustering result

        axs[j,i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, edgecolors='k')
        axs[j,i].set_title(f'{algo_name} - {dataset_name}')
        axs[j,i].set_title(f'{algo_name} - {dataset_name}')
        axs[j,i].set_xlabel('Feature 1')
        axs[j,i].set_ylabel('Feature 2')
        j=j+1
    i=i+1

# Add space between subplots
plt.subplots_adjust(hspace=1, wspace=0.5)
plt.show()

