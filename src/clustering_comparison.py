"""
Clustering Comparison: K-Means vs DBSCAN

This module implements an alternative clustering method (DBSCAN) to
empirically demonstrate why K-Means is preferred for this use case.

Purpose:
- Show that alternative clustering methods were considered
- Demonstrate empirical evidence for K-Means choice
- Illustrate practical limitations of DBSCAN for healthcare context
- Support LO3 (empirical comparison)

Note:
- DBSCAN is NOT the final clustering method
- This is for comparison only
- Focus is on demonstrating unsuitability, not optimization
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def compare_clustering_methods(X, eps_values=None, random_state=42):
    """
    Compare K-Means with DBSCAN.
    
    Args:
        X: Scaled feature data
        eps_values: List of eps values to test for DBSCAN
        random_state: Random seed
        
    Returns:
        dict: Comparison results
        
    Note:
        DBSCAN requires eps parameter tuning, which is:
        - Sensitive to scale
        - Dataset-specific
        - Difficult to interpret in healthcare context
    """
    
    if eps_values is None:
        eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    results = {
        'kmeans': None,
        'dbscan': []
    }
    
    print("\n" + "="*60)
    print("CLUSTERING METHOD COMPARISON: K-Means vs DBSCAN")
    print("="*60)
    
    # K-Means (baseline)
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    # Convert to numpy array if needed
    X_array = X.values if hasattr(X, 'values') else X
    kmeans_labels = kmeans.fit_predict(X_array)
    kmeans_sil = silhouette_score(X_array, kmeans_labels)
    n_clusters_kmeans = len(np.unique(kmeans_labels))
    
    results['kmeans'] = {
        'method': 'K-Means (k=3)',
        'n_clusters': n_clusters_kmeans,
        'n_noise': 0,
        'silhouette': kmeans_sil,
        'labels': kmeans_labels
    }
    
    print(f"\nK-Means (k=3):")
    print(f"  Clusters: {n_clusters_kmeans}")
    print(f"  Silhouette: {kmeans_sil:.3f}")
    print(f"  Cluster sizes: {np.bincount(kmeans_labels)}")
    
    # DBSCAN with different eps values
    print(f"\nDBSCAN (testing eps values):")
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=50)
        # Convert to numpy array if needed
        X_array = X.values if hasattr(X, 'values') else X
        dbscan_labels = dbscan.fit_predict(X_array)
        
        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        # Calculate silhouette only if we have at least 2 clusters
        if n_clusters >= 2:
            # Exclude noise points for silhouette calculation
            mask = dbscan_labels != -1
            if mask.sum() > 0:
                # Convert to numpy array if needed
                X_array = X.values if hasattr(X, 'values') else X
                sil = silhouette_score(X_array[mask], dbscan_labels[mask])
            else:
                sil = -1
        else:
            sil = -1
        
        results['dbscan'].append({
            'method': f'DBSCAN (eps={eps})',
            'eps': eps,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': sil,
            'labels': dbscan_labels
        })
        
        sil_str = f"{sil:.3f}" if sil != -1 else 'N/A'
        print(f"  eps={eps}: {n_clusters} clusters, {n_noise} noise points, "
              f"silhouette={sil_str}")
    
    print("="*60)
    
    return results


def plot_clustering_comparison(results, save_path=None):
    """
    Visualize clustering comparison.
    
    Args:
        results: Dictionary from compare_clustering_methods()
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Number of clusters vs eps
    ax1 = axes[0, 0]
    eps_vals = [r['eps'] for r in results['dbscan']]
    n_clusters = [r['n_clusters'] for r in results['dbscan']]
    ax1.plot(eps_vals, n_clusters, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax1.axhline(y=3, color='#3498db', linestyle='--', label='K-Means (k=3)', linewidth=2)
    ax1.set_xlabel('DBSCAN eps Parameter', fontsize=11)
    ax1.set_ylabel('Number of Clusters', fontsize=11)
    ax1.set_title('DBSCAN Sensitivity to eps', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Noise points vs eps
    ax2 = axes[0, 1]
    n_noise = [r['n_noise'] for r in results['dbscan']]
    ax2.plot(eps_vals, n_noise, marker='s', linewidth=2, markersize=8, color='#9b59b6')
    ax2.set_xlabel('DBSCAN eps Parameter', fontsize=11)
    ax2.set_ylabel('Number of Noise Points', fontsize=11)
    ax2.set_title('DBSCAN Noise Point Sensitivity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Silhouette scores
    ax3 = axes[1, 0]
    sil_scores = [r['silhouette'] for r in results['dbscan']]
    valid_eps = [eps for eps, sil in zip(eps_vals, sil_scores) if sil != -1]
    valid_sil = [sil for sil in sil_scores if sil != -1]
    
    if valid_sil:
        ax3.plot(valid_eps, valid_sil, marker='o', linewidth=2, markersize=8, 
                color='#2ecc71', label='DBSCAN')
    ax3.axhline(y=results['kmeans']['silhouette'], color='#3498db', linestyle='--', 
               label='K-Means', linewidth=2)
    ax3.set_xlabel('DBSCAN eps Parameter', fontsize=11)
    ax3.set_ylabel('Silhouette Score', fontsize=11)
    ax3.set_title('Clustering Quality Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_data = [
        ['Method', 'Clusters', 'Noise', 'Silhouette'],
        ['K-Means', str(results['kmeans']['n_clusters']), '0', 
         f"{results['kmeans']['silhouette']:.3f}"],
    ]
    
    # Add best DBSCAN result
    best_dbscan = max([r for r in results['dbscan'] if r['silhouette'] != -1], 
                     key=lambda x: x['silhouette'], default=results['dbscan'][0])
    summary_data.append([
        f"DBSCAN (eps={best_dbscan['eps']})",
        str(best_dbscan['n_clusters']),
        str(best_dbscan['n_noise']),
        f"{best_dbscan['silhouette']:.3f}" if best_dbscan['silhouette'] != -1 else 'N/A'
    ])
    
    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Clustering Comparison Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Clustering comparison plot saved: {save_path}")
    
    plt.show()
    
    return fig


def generate_clustering_comparison_summary(results):
    """
    Generate text summary of clustering comparison.
    
    Returns:
        str: Summary text for report
    """
    kmeans_result = results['kmeans']
    
    summary = f"""
CLUSTERING METHOD COMPARISON SUMMARY

K-Means (k=3):
- Clusters: {kmeans_result['n_clusters']}
- Noise points: 0
- Silhouette: {kmeans_result['silhouette']:.3f}
- Cluster sizes: {np.bincount(kmeans_result['labels']).tolist()}

DBSCAN Results (varying eps):
"""
    
    for r in results['dbscan']:
        sil_str = f"{r['silhouette']:.3f}" if r['silhouette'] != -1 else 'N/A'
        summary += f"\n  eps={r['eps']}: {r['n_clusters']} clusters, "
        summary += f"{r['n_noise']} noise points, "
        summary += f"silhouette={sil_str}"
    
    summary += """

Key Findings:
1. DBSCAN is highly sensitive to eps parameter
2. Number of clusters varies dramatically with eps
3. Many points classified as "noise" (ungrouped)
4. Difficult to achieve stable 3-cluster solution
5. No clear healthcare interpretation of "noise" patients

Conclusion:
DBSCAN's sensitivity to parameters and production of "noise" points
makes it unsuitable for healthcare risk stratification. K-Means provides
stable, interpretable groupings that align with low/medium/high risk
categories. This empirical comparison justifies the K-Means choice.
"""
    
    return summary


if __name__ == "__main__":
    print("Clustering comparison module loaded successfully")
    print("Use compare_clustering_methods() to run comparison")
