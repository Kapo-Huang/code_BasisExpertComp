import numpy as np
from sklearn.cluster import KMeans

def perform_kmeans_clustering(coords, K=20):
    """
    Paper Sec 3.1: "partitions the dataset... based on spatial coordinates"
    coords: (N, 4) tensor/array [x, y, z, t]
    Returns: List of indices for each cluster
    """
    # Only use x, y, z for clustering
    if hasattr(coords, 'cpu'):
        spatial_coords = coords[:, :3].cpu().numpy()
    else:
        spatial_coords = coords[:, :3]
        
    print(f"Executing K-Means with K={K}...")
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels = kmeans.fit_predict(spatial_coords)
    
    clusters = []
    for i in range(K):
        indices = np.where(labels == i)[0]
        if len(indices) > 0:
            clusters.append(indices)
            
    return clusters