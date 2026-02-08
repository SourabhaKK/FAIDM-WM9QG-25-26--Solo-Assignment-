"""
Clustering module using K-Means.

This module implements unsupervised learning to identify
population health profiles in the diabetes dataset.

Why K-Means?
- Interpretable: Creates clear group centroids
- Scalable: Works well with this dataset size
- Standard: Well-understood in health analytics

Limitations acknowledged:
- Assumes spherical clusters
- Sensitive to initialization
- Requires pre-specifying k
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def find_optimal_clusters(X, max_k=10, random_state=42):
    """
    Use elbow method to suggest number of clusters.
    
    Args:
        X (pd.DataFrame): Scaled feature data
        max_k (int): Maximum number of clusters to test
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Inertia values for different k
        
    Note:
        This is a guide, not a definitive answer.
        The "elbow" is often subjective in real data.
    """
    inertias = {}
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias[k] = kmeans.inertia_
    
    return inertias


def perform_clustering(X, n_clusters=3, random_state=42):
    """
    Perform K-Means clustering.
    
    Args:
        X (pd.DataFrame): Scaled feature data
        n_clusters (int): Number of clusters
        random_state (int): Random seed
        
    Returns:
        tuple: (fitted model, cluster labels, silhouette score)
        
    Note:
        n_clusters=3 is chosen as a reasonable starting point
        for health risk stratification (low/medium/high risk).
        This is a practical choice, not an optimized one.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Evaluate using silhouette score
    # This measures how well-separated clusters are
    sil_score = silhouette_score(X, cluster_labels)
    
    print(f"Clustering completed with {n_clusters} clusters")
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Cluster sizes: {np.bincount(cluster_labels)}")
    
    return kmeans, cluster_labels, sil_score


def analyze_cluster_characteristics(X, cluster_labels, feature_names=None):
    """
    Analyze characteristics of each cluster.
    
    Args:
        X (pd.DataFrame): Feature data (scaled)
        cluster_labels (np.array): Cluster assignments
        feature_names (list, optional): Names of features
        
    Returns:
        pd.DataFrame: Mean feature values per cluster
        
    Note:
        This helps interpret what each cluster represents.
        However, clusters may not have clear clinical meaning.
    """
    if feature_names is None:
        feature_names = X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1])
    
    # Create DataFrame with cluster labels
    df_clustered = pd.DataFrame(X, columns=feature_names)
    df_clustered['Cluster'] = cluster_labels
    
    # Calculate mean values per cluster
    cluster_profiles = df_clustered.groupby('Cluster').mean()
    
    return cluster_profiles


def plot_elbow_curve(inertias, save_path=None):
    """
    Plot elbow curve for cluster selection.
    
    Args:
        inertias (dict): Inertia values from find_optimal_clusters
        save_path (str, optional): Path to save figure
        
    Note:
        The elbow point is often ambiguous.
        This plot is a guide, not a prescription.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(list(inertias.keys()), list(inertias.values()), marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved: {save_path}")
    
    plt.show()
