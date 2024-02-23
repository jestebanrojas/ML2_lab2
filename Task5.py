
"""
--------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
5. Let’s use the newly created modules in unsupervised to cluster some toy data.
"""

"""
--------------------------------------------------------------------------------------
a. Use the following code snippet to create scattered data X
from sklearn.datasets import make_blobs
X, y = make_blobs(
n_samples=500,
n_features=2,
centers=4,
cluster_std=1,
center_box=(-10.0, 10.0),
shuffle=True,
random_state=1,
)
--------------------------------------------------------------------------------------
"""


from sklearn.datasets import make_blobs
from unsupervised.clustering import kmeans
from unsupervised.clustering import kmedoids
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_samples, silhouette_score


X, y = make_blobs(
n_samples=500,
n_features=2,
centers=4,
cluster_std=1,
center_box=(-10.0, 10.0),
shuffle=True,
random_state=1,
)


"""
--------------------------------------------------------------------------------------
b. Plot the resulting dataset. How many clusters are there? How far are they from one another?
--------------------------------------------------------------------------------------
"""

# Define the range of clusters to iterate over
n_clusters_range = range(2, 6)

# Plot the clusters for both KMeans and KMedoids on a grid
fig, axs = plt.subplots(len(n_clusters_range)+1, 2, figsize=(24, 20))
# Add space between subplots
plt.subplots_adjust(hspace=1.4)
fig_silhouete, axs_silhouete = plt.subplots(len(n_clusters_range), 2, figsize=(24, 10))
# Add space between subplots
plt.subplots_adjust(hspace=1)


# Scatter plot of the generated data
axs[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
axs[0, 0].set_title(f'Generated Data with Clusters')
axs[0, 0].set_xlabel('Feature 1')
axs[0, 0].set_ylabel('Feature 2')

axs[0, 1].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
axs[0, 1].set_title(f'Generated Data with Clusters')
axs[0, 1].set_xlabel('Feature 1')
axs[0, 1].set_ylabel('Feature 2')

# Find unique cluster labels
unique_labels = np.unique(y)

# Calculate cluster centers
cluster_centers = []
for label in unique_labels:
    cluster_centers.append(np.mean(X[y == label], axis=0))

print("Cluster Centers:")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i + 1}: {center}")


# Compute pairwise distances between centroids
cluster_distances = euclidean_distances(cluster_centers)

# Display distance matrix
print("Distance Matrix between Cluster Centroids:")
print(cluster_distances)

"""
There are 4 clusters. The distance (using euclidean distance) between clusters is as follows:

Cluster 1 - Cluster 2: 11.875
Cluster 1 - Cluster 3: 13.683
Cluster 1 - Cluster 4: 8.888
Cluster 2 - Cluster 3: 5.114
Cluster 2 - Cluster 4: 3.936
Cluster 3 - Cluster 4: 4.997

"""
"""
--------------------------------------------------------------------------------------
c. For both k-means and k-medoids (your implementations), calculate the silhouette plots and
coefficients for each run, iterating K from 1 to 5 clusters.
--------------------------------------------------------------------------------------
"""

for i, n_clusters in enumerate(n_clusters_range):
    # KMeans
    #k_means=kmeans.KMeans(n_clusters=2)
    k_means = kmeans.KMeans(n_clusters=n_clusters)
    k_means.fit(X)
    axs[i+1, 0].scatter(X[:, 0], X[:, 1], c=k_means.labels_, cmap='viridis', edgecolor='k', s=50)
    axs[i+1, 0].set_title(f'KMeans Clustering with {n_clusters} clusters')
    axs[i+1, 0].set_xlabel('Feature 1')
    axs[i+1, 0].set_ylabel('Feature 2')

    # Calcular la puntuación de la silueta para cada muestra
    silhouette_avg = silhouette_score(X, k_means.labels_)
    sample_silhouette_values = silhouette_samples(X, k_means.labels_)


    axs_silhouete[i, 0].set_title(f"Clústeres = {n_clusters}\nCoeficiente de silueta = {silhouette_avg:.2f}")
    y_lower = 10
    for j in range(n_clusters):
        # Agregar la silueta para cada muestra
        ith_cluster_silhouette_values = sample_silhouette_values[k_means.labels_ == j]
        ith_cluster_silhouette_values.sort()

        size_cluster_j = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_j

        color = plt.cm.viridis(float(j) / n_clusters)
        axs_silhouete[i, 0].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)


        # Etiquetas y límites para el siguiente cluster
        y_lower = y_upper + 10  

    axs_silhouete[i, 0].axvline(x=silhouette_avg, color="red", linestyle="--")


    # KMedoids
    k_medoids = kmedoids.KMedoids(n_clusters=n_clusters)
    k_medoids.fit(X)
    axs[i+1, 1].scatter(X[:, 0], X[:, 1], c=k_medoids.labels_, cmap='viridis', edgecolor='k', s=50)
    axs[i+1, 1].set_title(f'KMedoids Clustering with {n_clusters} clusters')
    axs[i+1, 1].set_xlabel('Feature 1')
    axs[i+1, 1].set_ylabel('Feature 2')
    
    # Calcular la puntuación de la silueta para cada muestra
    silhouette_avg = silhouette_score(X, k_medoids.labels_)
    sample_silhouette_values = silhouette_samples(X, k_medoids.labels_)

    y_lower = 10
    axs_silhouete[i, 1].set_title(f"Clústeres = {n_clusters}\nCoeficiente de silueta = {silhouette_avg:.2f}")
    for j in range(n_clusters):
        # Agregar la silueta para cada muestra
        ith_cluster_silhouette_values = sample_silhouette_values[k_medoids.labels_ == j]
        ith_cluster_silhouette_values.sort()

        size_cluster_j = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_j

        color = plt.cm.viridis(float(j) / n_clusters)
        axs_silhouete[i, 1].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)


        # Etiquetas y límites para el siguiente cluster
        y_lower = y_upper + 10  

    axs_silhouete[i, 1].axvline(x=silhouette_avg, color="red", linestyle="--")


plt.yticks([])  # Borrar las etiquetas de los clusters
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.tight_layout()

# Add space between subplots
plt.subplots_adjust(hspace=1, wspace=0.5)
plt.show()

