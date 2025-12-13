import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans



def k_center_greedy(X, k): ## try other way
    """
    Core-set selection via Farthest-First Traversal (K-center greedy)
    """
    n = len(X)
    # 1) случайная стартовая точка
    centers = [np.random.randint(0, n)]

    # 2) расстояния до первого центра
    dist = pairwise_distances(X, X[centers], metric="euclidean")

    for _ in range(k - 1):
        # берём точку, которая дальше всех от ближайшего центра
        idx = np.argmax(dist.min(axis=1))
        centers.append(idx)

        # обновляем расстояния
        new_dist = pairwise_distances(X, X[[idx]])
        dist = np.hstack((dist, new_dist))

    return centers


def diversity_sampling(X, k):
    n = len(X)
    selected = [np.random.randint(0, n)]
    dist = pairwise_distances(X, X[selected]).min(axis=1)
    
    for _ in range(k-1):
        idx = np.argmax(dist)
        selected.append(idx)
        new_dist = pairwise_distances(X, X[[idx]]).min(axis=1)
        dist = np.minimum(dist, new_dist)
    return selected


def kmeans_coreset(X, k):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X)
    # выбираем ближайшую реальную точку к каждому центру
    centers = km.cluster_centers_
    indices = []
    for c in centers:
        idx = np.argmin(np.sum((X - c)**2, axis=1))
        indices.append(idx)
    return indices


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="path to TRAIN CSV")
    args = parser.parse_args()

    # === Load dataset ===
    df = pd.read_csv(args.file)

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # === Normalize ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === Desired subset size (10% of dataset) ===
    fraction = 0.1
    k = int(len(X_scaled) * fraction)
    print(f"Selecting {k} points out of {len(X_scaled)}...")

    # === Core-Set K-Center Selection ===
    

    # Select indices
    # subset_indices = k_center_greedy(X_scaled, k)
    subset_indices = diversity_sampling(X_scaled, k)

    # Extract full rows
    df_subset = df.iloc[subset_indices].reset_index(drop=True)

    # Save subset as CSV
    df_subset.to_csv(f"subset.csv", index=False)

if __name__ == "__main__":
    main()
