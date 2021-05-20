import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs


K = 3
MAX_ITERS = 150
PLOT_STEPS = True
CLUSTERS = [[] for _ in range(K)]
CENTROIDS = []
N_SAMPLES = 8
N_FEATURES = 2

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def predict(X):
    X = X
   
    N_SAMPLES, N_FEATURES = X.shape
   
    # initialize
    CENTROIDS = [[2,10],[5,8],[1,2]]

    # Optimize CLUSTERS
    iteration_count = 1
    for _ in range(MAX_ITERS):
        # Assign samples to closest CENTROIDS (create CLUSTERS)
        print(f"Iteration {iteration_count}\n")
        CLUSTERS = _create_CLUSTERS(CENTROIDS)
        print(CENTROIDS)
        for i in range(3):
            example_numbers = ",".join(map(str, CLUSTERS[i]))
            centroid_ = ",".join(map(str, CENTROIDS[i]))
            print(f"Cluster {i}: {example_numbers}")           
            print(f"Centroid: ({centroid_})\n")
        iteration_count += 1

        print("")
        # if PLOT_STEPS:
        #     plot()

        # Calculate new CENTROIDS from the CLUSTERS
        CENTROIDS_old = CENTROIDS
        CENTROIDS = _get_CENTROIDS(CLUSTERS)

        # check if CLUSTERS have changed
        if _is_converged(CENTROIDS_old, CENTROIDS):
            break

        # if PLOT_STEPS:
        #     plot()
       

    # Classify samples as the index of their CLUSTERS
    return _get_cluster_labels(CLUSTERS)


def _get_cluster_labels(CLUSTERS):
    # each sample will get the label of the cluster it was assigned to
    labels = np.empty(N_SAMPLES)

    for cluster_idx, cluster in enumerate(CLUSTERS):
        for sample_index in cluster:
            labels[sample_index] = cluster_idx
    return labels

def _create_CLUSTERS(CENTROIDS):
    # Assign the samples to the closest CENTROIDS to create CLUSTERS
    CLUSTERS = [[] for _ in range(K)]
    for idx, sample in enumerate(X):
        centroid_idx = _closest_centroid(sample, CENTROIDS)
        CLUSTERS[centroid_idx].append(idx)
    return CLUSTERS

def _closest_centroid(sample, CENTROIDS):
    # distance of the current sample to each centroid
    distances = [euclidean_distance(sample, point) for point in CENTROIDS]
    closest_index = np.argmin(distances)
    return closest_index

def _get_CENTROIDS(CLUSTERS):
    # assign mean value of CLUSTERS to CENTROIDS
    CENTROIDS = np.zeros((K, N_FEATURES))
    for cluster_idx, cluster in enumerate(CLUSTERS):
        cluster_mean = np.mean(X[cluster], axis=0)
        CENTROIDS[cluster_idx] = cluster_mean
    return CENTROIDS

def _is_converged(CENTROIDS_old, CENTROIDS):
    # distances between each old and new CENTROIDS, fol all CENTROIDS
    distances = [euclidean_distance(CENTROIDS_old[i], CENTROIDS[i]) for i in range(K)]
    return sum(distances) == 0

def plot():
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, index in enumerate(CLUSTERS):
        point = X[index].T
        ax.scatter(*point)

    for point in CENTROIDS:
        ax.scatter(*point, marker="x", color='black', linewidth=2)

    plt.show()


X = [
    [2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]
]
X = np.array(X)  
# X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)

    
# clusters = len(np.unique(y))


predict(X)