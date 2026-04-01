#!/usr/bin/env python3
"""Shared utility functions for embedding visualization scripts."""

import numpy as np
from sklearn.decomposition import PCA
from openai import OpenAI
from matplotlib.ticker import MultipleLocator

# Try to import UMAP (optional dependency)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Color definitions for plots
COLORS = {
    'ar_only': '#0000FF',         # Blue
    'cross_only': '#FF0000',      # Red
    'no_only': '#FFFF00',         # Yellow
    'ar_cross': '#8000FF',        # Purple
    'cross_no': '#00FF00',        # Green
    'ar_no': '#FF8000',           # Orange
    'all_three': '#8B4513'        # Brown
}

def generate_embeddings(texts, api_key, batch_size=2048):
    """Generate embeddings for a list of texts using OpenAI's API.

    Args:
        texts: List of text strings to embed
        api_key: OpenAI API key
        batch_size: Maximum number of texts per API call (OpenAI limit is 2048)

    Returns:
        Tuple of (embeddings, total_tokens, estimated_cost)
        - embeddings: np.array of shape (n_texts, embedding_dim)
        - total_tokens: int, total tokens used
        - estimated_cost: float, estimated cost in USD
    """
    print(f"  Generating embeddings for {len(texts)} texts")

    client = OpenAI(api_key=api_key)

    all_embeddings = []
    total_tokens = 0

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if len(texts) > batch_size:
            print(f"    Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}: {len(batch)} texts")

        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        total_tokens += response.usage.total_tokens

    embeddings = np.array(all_embeddings)
    estimated_cost = total_tokens / 1_000_000 * 0.020

    return embeddings, total_tokens, estimated_cost

def apply_pca(embeddings, random_state=42):
    """Apply PCA to reduce embeddings to 2D.

    Args:
        embeddings: np.array of shape (n_samples, n_features)
        random_state: Random state for reproducibility

    Returns:
        Tuple of (embeddings_2d, explained_variance)
        - embeddings_2d: np.array of shape (n_samples, 2)
        - explained_variance: np.array of shape (2,) with explained variance ratios
    """
    pca = PCA(n_components=2, random_state=random_state)
    embeddings_2d = pca.fit_transform(embeddings)
    explained_variance = pca.explained_variance_ratio_
    return embeddings_2d, explained_variance

def apply_umap(embeddings, random_state=42, n_neighbors=15, min_dist=0.1):
    """Apply UMAP to reduce embeddings to 2D.

    Args:
        embeddings: np.array of shape (n_samples, n_features)
        random_state: Random state for reproducibility
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter

    Returns:
        Tuple of (embeddings_2d, None)
        - embeddings_2d: np.array of shape (n_samples, 2)
        - None: UMAP doesn't have explained variance (for consistency with PCA)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")

    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        transform_seed=random_state,
        low_memory=True
    )
    embeddings_2d = reducer.fit_transform(embeddings)
    return embeddings_2d, None

def calculate_pairwise_distances(embeddings):
    """Calculate all pairwise Euclidean distances between embeddings.

    Args:
        embeddings: np.array of shape (n_samples, n_features)

    Returns:
        np.array of pairwise distances
    """
    n = len(embeddings)
    if n < 2:
        return np.array([])

    from scipy.spatial.distance import pdist
    pairwise_distances = pdist(embeddings, metric='euclidean')
    return pairwise_distances

def add_grid_styling(ax, method='pca'):
    """Add granular gridlines and styling to matplotlib axes.

    Args:
        ax: matplotlib axes object
        method: 'pca' or 'umap', affects tick intervals and label size
    """
    if method.lower() == 'umap':
        # UMAP has larger value ranges, use automatic tick locators
        # with reduced number of ticks to avoid overlap
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
        # No minor ticks for UMAP to keep it clean
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.4)
        ax.tick_params(axis='both', which='major', labelsize=8, length=6)
    else:
        # PCA: use fine-grained ticks every 0.1 (major) and 0.05 (minor)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.4)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.2)
        ax.tick_params(axis='both', which='major', labelsize=10, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=8, length=3)

def calculate_similarity_matrix(embeddings, kernel='rbf', sigma=None):
    """Calculate similarity matrix from embeddings using specified kernel.

    Args:
        embeddings: np.array of shape (n_samples, n_features)
        kernel: 'rbf' or 'cosine'
        sigma: RBF kernel bandwidth (if None, uses median heuristic)

    Returns:
        Similarity matrix (n_samples x n_samples)
    """
    n = len(embeddings)
    if n < 2:
        return np.eye(1)

    if kernel == 'cosine':
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        # Cosine similarity matrix
        K = normalized @ normalized.T
    elif kernel == 'rbf':
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform
        pairwise_dists = pdist(embeddings, metric='euclidean')

        # Median heuristic for sigma if not provided
        if sigma is None:
            sigma = np.median(pairwise_dists)
            if sigma == 0:
                sigma = 1.0  # Fallback if all embeddings are identical

        # RBF kernel: K[i,j] = exp(-||x_i - x_j||^2 / (2*sigma^2))
        dist_matrix = squareform(pairwise_dists)
        K = np.exp(-dist_matrix**2 / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Use 'rbf' or 'cosine'.")

    return K

def calculate_vendi_score(embeddings, kernel='rbf', sigma=None):
    """Calculate Vendi score (entropy-based diversity metric) from embeddings.

    Args:
        embeddings: np.array of shape (n_samples, n_features)
        kernel: 'rbf' or 'cosine'
        sigma: RBF kernel bandwidth (if None, uses median heuristic)

    Returns:
        Dict with:
            - vendi_score: float, exponential of entropy
            - entropy: float, Shannon entropy of eigenvalues
            - eigenvalues: np.array, eigenvalues of normalized similarity matrix
            - kernel: str, kernel used
            - sigma: float, sigma parameter (for RBF)
    """
    n = len(embeddings)

    # Edge case: < 2 samples
    if n < 2:
        return {
            'vendi_score': 1.0,
            'entropy': 0.0,
            'eigenvalues': np.array([1.0]),
            'kernel': kernel,
            'sigma': sigma
        }

    # Calculate similarity matrix
    K = calculate_similarity_matrix(embeddings, kernel=kernel, sigma=sigma)

    # Normalize by trace
    trace = np.trace(K)
    if trace == 0:
        trace = 1.0  # Fallback
    K_norm = K / trace

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigh(K_norm)[0]

    # Filter positive eigenvalues (for numerical stability)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Calculate Shannon entropy
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))

    # Vendi score
    vendi_score = np.exp(entropy)

    return {
        'vendi_score': float(vendi_score),
        'entropy': float(entropy),
        'eigenvalues': eigenvalues,
        'kernel': kernel,
        'sigma': sigma if kernel == 'rbf' else None
    }

def calculate_vendi_scores_multiple_kernels(embeddings):
    """Calculate Vendi scores with multiple kernel configurations.

    Args:
        embeddings: np.array of shape (n_samples, n_features)

    Returns:
        Dict mapping kernel name to vendi_score result
    """
    results = {}

    # Cosine kernel
    results['cosine'] = calculate_vendi_score(embeddings, kernel='cosine')

    # RBF with auto sigma (median heuristic)
    results['rbf_auto'] = calculate_vendi_score(embeddings, kernel='rbf', sigma=None)

    # Get median for other sigmas
    from scipy.spatial.distance import pdist
    if len(embeddings) >= 2:
        pairwise_dists = pdist(embeddings, metric='euclidean')
        median_dist = np.median(pairwise_dists)

        # RBF with 0.5x median
        results['rbf_0.5x'] = calculate_vendi_score(embeddings, kernel='rbf', sigma=0.5 * median_dist)

        # RBF with 2.0x median
        results['rbf_2.0x'] = calculate_vendi_score(embeddings, kernel='rbf', sigma=2.0 * median_dist)

    return results
