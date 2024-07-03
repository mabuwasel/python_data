import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Function to determine the optimal number of clusters using Silhouette Score
def optimal_k(X: pd.DataFrame,max_k: int = 10, visualize: bool = True):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    wcss = []
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    
    # visualizing Silhouette Score
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', linestyle='--')
        plt.title('Silhouette Score Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.show()

    # Choose the optimal k with the highest silhouette score
    optimal_k = np.argmax(silhouette_scores) + 2  # adding 2 because the range starts at 2
    return optimal_k




