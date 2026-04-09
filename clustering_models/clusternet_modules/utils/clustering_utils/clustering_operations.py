#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
from kmeans_pytorch import kmeans as GPU_KMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
#import inspect
from ..torch_clustering import PyTorchKMeans
import torch.nn.functional as F 
import torch.nn as nn
## init_mus_and_covs sera appelé qu'une fois a l'epoch 0 donc  la seul option qu'on utilisera ici est how_to_init_mu kmeans (qui sera ici d'ailleurs spherical kmeans)
from torch.distributions.constraints import positive_definite
from datetime import datetime
from sklearn.metrics import silhouette_score,davies_bouldin_score
import numpy as np 
import math
import random


def _safe_silhouette(X: np.ndarray, labels: np.ndarray, metric: str = 'cosine') -> float:
    """Return silhouette score or -1.0 when invalid (n<3 or only one label)."""
    if X.shape[0] < 3:
        return -1.0
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return -1.0
    return silhouette_score(X, labels, metric=metric)

def init_mus_and_covs(codes, K, how_to_init_mu, logits, use_priors=True, prior=None, random_state=0, device="cpu", prior_choice='data_std'):
    """
    This function initializes the clusters' centers and covariance matrices.

    Args:
        codes (torch.tensor): The codes that should be clustered, in R^{N x D}.
        how_to_init_mu (str): A string defining how to initialize the centers.
        use_priors (bool, optional): Whether to consider the priors. Defaults to True.
        prior (object, optional): A prior object with functions for computing posteriors.
        random_state (int, optional): Random seed for reproducibility.
        device (str, optional): The device to run on.
        prior_choice (str, optional): Choice of prior initialization ('data_std' or 'dynamic_data_std').
    """
    print("Initializing clusters params using {}...".format(how_to_init_mu))
    
    # For large datasets, sample only a portion of the codes.
    if codes.shape[0] > 2 * (10 ** 5):
        codes = codes[:2 * (10**5)]
    
    if how_to_init_mu == "kmeans":
        # Use PyTorchKMeans with cosine metric for clustering.
        #TO REMOVE 
        random_state = random.randint(0,10000000)
        #
        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': random_state,
            'n_clusters': K,
            'verbose': True
        }
        clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        labels = clustering_model.fit_predict(codes.detach())
        
        # Get the unique cluster labels and counts.
        number, counts = torch.unique(labels, return_counts=True)
        
        # Instead of directly using kmeans_mus, compute the cluster centers with KarcherMean.
        mus = torch.stack([
            KarcherMean(soft_assign=None, codes=codes[torch.where(labels == i)]).to(device=device)
            for i in number
        ])
        
        # Compute the mixing proportions.
        pi = counts / float(len(codes))
        
        # Compute the data covariance matrices via hard assignment.
        data_covs = compute_data_covs_hard_assignment(labels, codes, K, mus.cpu(), prior)
        D = len(codes[0])

        # If using priors, update the covariances accordingly.
        if use_priors:
            covs = []
            if prior_choice == 'dynamic_data_std':
                for k in range(K):
                    cov_k = prior.compute_post_cov(counts[k], data_covs[k], D, psi_index=k)
                    covs.append(cov_k)
            else:
                for k in range(K):
                    cov_k = prior.compute_post_cov(counts[k], data_covs[k], D)
                    covs.append(cov_k)
            covs = torch.stack(covs)
        else:
            # Empirical covariances — ensure PD
            covs = []
            for k in range(K):
                if not positive_definite.check(data_covs[k]):
                    print(f"[WARNING] Covariance matrix at cluster {k} is not PD. Adjusting...")
                    cov_k = ensure_positive_definite(data_covs[k])
                else:
                    cov_k = data_covs[k]
                covs.append(cov_k)
            covs = torch.stack(covs)

            
        return mus, covs, pi, labels, prior

    elif how_to_init_mu == "umap":
        import umap  # ensure that umap is imported
        
        # Obtain a UMAP embedding of the codes.
        n_neighbors = min(30, len(codes))
        init_method = 'random' if n_neighbors >= len(codes) else 'spectral'
        umap_obj = umap.UMAP(
            init=init_method,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=3,
            #random_state=42,
            metric='cosine'
        ).fit(codes.detach().cpu())
        umap_codes = umap_obj.embedding_
        print('UMAP SHAPE:', umap_codes.shape)
        
        kwargs = {
            'metric': 'euclidean',
            'distributed': False,
            #'random_state': random_state,
            'n_clusters': K,
            'verbose': True
        }
        clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        labels = clustering_model.fit_predict(torch.from_numpy(umap_codes).detach())
        
        number, counts = torch.unique(labels, return_counts=True)
        
        # Compute the cluster centers using KarcherMean.
        mus = torch.stack([
            KarcherMean(soft_assign=None, codes=codes[torch.where(labels == i)]).to(device=device)
            for i in number
        ])
        
        print('prior_choice:', prior_choice)
        if prior_choice == 'dynamic_data_std':
            print('INTO dynamic data std init_mus_covs')
            prior.init_priors(codes.detach().cpu(), labels)
        
        pi = counts / float(len(codes))
        data_covs = compute_data_covs_hard_assignment(labels, codes, K, mus.cpu(), prior)
        D = len(codes[0])
        
        if use_priors:
            covs = []
            if prior_choice == 'dynamic_data_std':
                for k in range(K):
                    cov_k = prior.compute_post_cov(counts[k], data_covs[k], D, psi_index=k)
                    covs.append(cov_k)
            else:
                for k in range(K):
                    cov_k = prior.compute_post_cov(counts[k], data_covs[k], D)
                    covs.append(cov_k)
            covs = torch.stack(covs)
        else:
            covs = []
            for k in range(K):
                cov_k = data_covs[k]
                if not positive_definite.check(cov_k):
                    print(f"[WARNING] Covariance matrix at cluster {k} is not positive definite. Fixing...")
                    cov_k = ensure_positive_definite(cov_k)
                covs.append(cov_k)
            covs = torch.stack(covs)

        
        # Optional: Save debugging files (paths can be adjusted as needed)
        # torch.save(umap_codes, '/path/to/debug/init_umap_codes.pt')
        # torch.save(codes, '/path/to/debug/init_codes.pt')
        # torch.save(labels, '/path/to/debug/init_labels.pt')
        
        return mus, covs, pi, labels, prior
    else:
        raise ValueError("Unknown initialization method: {}".format(how_to_init_mu))

        
def init_mus_and_covs_DPM(codes, K, how_to_init_mu, logits, use_priors=True, prior=None, random_state=0, device="cpu"):
    """This function initalizes the clusters' centers and covariances matrices.

    Args:
        codes (torch.tensor): The codes that should be clustered, in R^{N x D}.
        how_to_init_mu (str): A string defining how to initialize the centers.
        use_priors (bool, optional): Whether to consider the priors. Defaults to True.
    """
    print("Initializing clusters params using Kmeans...")
    if codes.shape[0] > 2 * (10 ** 5):
        # sample only a portion of the codes
        
        codes = codes[:2 * (10**5)]
    if how_to_init_mu == "kmeans":
        if K == 1:
            kmeans = KMeans(n_clusters=K, random_state=random_state).fit(codes.detach().cpu())
            labels = torch.from_numpy(kmeans.labels_)
            kmeans_mus = torch.from_numpy(kmeans.cluster_centers_)
        else:
            labels, kmeans_mus = GPU_KMeans(X=codes.detach(), num_clusters=K, device=device)
        number, counts = torch.unique(labels, return_counts=True)
        print('INTO init_mus_and_covs Kmeans initialisation number of K found:',number)
        print('number of codes clusterized :', codes.size())
        print('number of clusters specified:  ',K)
        print(' number labels per cluster :',counts)
        pi = counts / float(len(codes))
        data_covs = f_hard_assignment(labels, codes, K, kmeans_mus.cpu(), prior)

        if use_priors:
            mus = prior.compute_post_mus(counts, kmeans_mus.cpu())
            covs = []
            for k in range(K):
                codes_k = codes[labels == k]
                cov_k = prior.compute_post_cov(counts[k], codes_k.mean(axis=0), data_covs[k])  # len(codes_k) == counts[k]? yes
                covs.append(cov_k)
            covs = torch.stack(covs)
        else:
            mus = kmeans_mus
            covs = data_covs
        return mus, covs, pi, labels

    elif how_to_init_mu == "kmeans_1d":
        pca = PCA(n_components=1)
        pca_codes = pca.fit_transform(codes.detach().cpu())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes).to(device=device), num_clusters=K, device=torch.device(device))

        kmeans_mus = torch.tensor(
            pca.inverse_transform(cluster_centers.cpu().numpy()),
            device=device,
            requires_grad=False,
        )
        _, counts = torch.unique(torch.tensor(labels), return_counts=True)
        pi = counts / float(len(codes))
        data_covs = compute_data_covs_hard_assignment(labels, codes, K, kmeans_mus.cpu(), prior)

        if use_priors:
            mus = prior.compute_post_mus(counts, kmeans_mus.cpu())
            covs = []
            for k in range(K):
                codes_k = codes[labels == k]
                cov_k = prior.compute_post_cov(counts[k], codes_k.mean(axis=0), data_covs[k])  # len(codes_k) == counts[k]? yes
                covs.append(cov_k)
            covs = torch.stack(covs)
        else:
            mus = kmeans_mus
            covs = data_covs

        return mus, covs, pi, labels

    elif how_to_init_mu == "soft_assign":
        mus = compute_mus_soft_assignment(codes, logits, K)
        pi = compute_pi_k(logits, prior=prior if use_priors else None)
        data_covs = compute_data_covs_soft_assignment(logits, codes, K, mus.cpu(), prior.name)

        if use_priors:
            mus = prior.compute_post_mus(pi, mus)
            covs = []
            for k in range(K):
                r_k = pi[k] * len(codes)  # if it the sum of logits change to this becuase this is confusing
                cov_k = prior.compute_post_cov(r_k, mus[k], data_covs[k])
                covs.append(cov_k)
            covs = torch.stack(covs)
        else:
            covs = data_covs
        return mus, covs, pi, logits.argmax(axis=1)

"""
from datetime import datetime
import torch
import umap
from sklearn.decomposition import PCA"""
# (Make sure PyTorchKMeans, silhouette_score, KarcherMean, and compute_data_covs_hard_assignment are imported or defined)

from datetime import datetime
import torch
import torch.nn.functional as F
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# Ensure that PyTorchKMeans, silhouette_score, KarcherMean, compute_data_covs_hard_assignment,
# Log_mapping, and Exp_mapping are imported or defined appropriately.

def init_mus_and_covs_sub( 
    codes, 
    k, 
    mus, 
    n_sub, 
    how_to_init_mu_sub, 
    logits, 
    logits_sub, 
    prior=None, 
    use_priors=True, 
    random_state=0, 
    device="cpu",
    fixed_subclusters=False  # Control fixed vs. dynamic subclusters
):
    """
    Initialize subcluster means (mus_sub), covariances (covs_sub), and mixing coefficients (pi_sub).

    Parameters:
      - codes: Tensor of shape [num_samples, code_dim]
      - k: Index of the main cluster being processed
      - mus: Tensor of main cluster means
      - n_sub: Desired number of subclusters
      - how_to_init_mu_sub: Method to initialize subcluster means ('kmeans', 'umap', 'kmeans_1d', or 'soft_assign')
      - logits: Tensor of shape [num_samples, K] representing main cluster assignments
      - logits_sub: Tensor of shape [num_samples, total_subclusters] representing subcluster assignments
      - prior: Prior object for computing posterior covariances
      - use_priors: Boolean indicating whether to use priors
      - random_state: Seed for reproducibility
      - device: Device to perform computations on ('cpu' or 'cuda')
      - fixed_subclusters: If True, initializes with exactly n_sub subclusters (no dynamic adjustment)
    """
    print("Initializing mus_and_covs_sub")
    
    if how_to_init_mu_sub == "kmeans":
        indices_k = logits.argmax(-1) == k
        codes_k = codes[indices_k]
        
        # If there are too few datapoints, create ghost subclusters.
        if len(codes_k) <= n_sub: 
            print('Empty or insufficient cluster:', codes_k.size())
            D = codes.size(1)
            mus_sub = torch.zeros((n_sub, D), device=codes.device)
            covs_sub = torch.stack([torch.eye(D, device=codes.device) for _ in range(n_sub)])
            pi_sub = torch.zeros(n_sub, device=codes.device)
            return mus_sub, covs_sub, pi_sub, n_sub
        
        else:
            # Directly use the raw data without dimensionality reduction.
            reduced_codes = codes_k.detach().cpu().numpy()
            print('Using raw codes for clustering. Shape:', reduced_codes.shape)
            
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # --------------------------
            # Clustering & Subcluster Initialization
            # --------------------------
            if fixed_subclusters:
                # Fixed subclusters: directly cluster into n_sub groups.
                n_clusters = n_sub
                print(f"Fixed subclusters: Initializing with {n_clusters} subclusters.")
                clustering_model = PyTorchKMeans(
                    init='k-means++', 
                    max_iter=300, 
                    tol=1e-4, 
                    n_clusters=n_clusters, 
                    metric='cosine',
                    random_state=random_state,
                    verbose=False
                )
                labels = clustering_model.fit_predict(torch.from_numpy(reduced_codes).to(device=device))
    
                #silhouette_avg = silhouette_score(
                #    torch.from_numpy(reduced_codes).cpu(), 
                #    labels.cpu(), 
                #    metric='cosine'
                #)
                #print(f"Silhouette Score for fixed {n_clusters} clusters: {silhouette_avg}")
    
                # Compute the subcluster means using the Karcher mean (over the original codes).
                mus_sub = torch.stack([
                    KarcherMean(soft_assign=None, 
                                codes=codes_k[torch.where(labels == i)[0].to(device=codes_k.device)],
                                tol=1e-5, max_iter=100
                               ).to(device=codes_k.device) 
                    for i in range(n_clusters)
                ])
                print(f'Initialized mus_sub shape: {mus_sub.shape}')
    
                # Compute covariance estimates with hard assignments.
                data_covs_sub = compute_data_covs_hard_assignment(
                    labels.to(device), 
                    codes_k.to(device), 
                    n_clusters, 
                    mus_sub, 
                    prior
                )
                counts = torch.tensor([
                    (labels == i).sum().item() for i in range(n_clusters)
                ], dtype=torch.float32, device=codes.device)
                if use_priors:
                    covs_sub = []
                    D = codes.size(1)
                    for i in range(n_clusters):
                        if prior.prior_choice != 'dynamic_data_std':
                            cov_k = prior.compute_post_cov(counts[i], data_covs_sub[i], D)
                        else:
                            cov_k = prior.compute_post_cov(counts[i], data_covs_sub[i], D, k)
                        covs_sub.append(cov_k.to(codes.device))
                else:
                    covs_sub = []
                    D = codes.size(1)
                    for i in range(n_clusters):
                        cov_k = data_covs_sub[i]
                        if not positive_definite.check(cov_k):
                            print(f"[WARNING] Covariance matrix at cluster {i} is not positive definite. Adjusting...")
                            cov_k = ensure_positive_definite(cov_k)
                        covs_sub.append(cov_k.to(codes.device))

    
                pi_sub = counts / float(len(codes))
    
                return mus_sub, covs_sub, pi_sub, n_clusters
            
            else:
                # Dynamic subcluster initialization: search over 2 to n_sub clusters.
                best_silhouette = -1
                best_mus_sub = None
                best_covs_sub = None
                best_pi_sub = None
                best_n_clusters = 2
                best_labels = None
    
                for n_clusters in range(2, n_sub + 1):
                    print(f"Trying {n_clusters} subclusters for cluster {k}.")
                    clustering_model = PyTorchKMeans(
                        init='k-means++', 
                        max_iter=300, 
                        tol=1e-4, 
                        n_clusters=n_clusters, 
                        metric='cosine',
                        random_state=random_state,
                        verbose=False
                    )
                    labels = clustering_model.fit_predict(torch.from_numpy(reduced_codes).to(device=device))
                    silhouette_avg = _safe_silhouette(torch.from_numpy(reduced_codes).cpu(), labels.cpu(),metric='cosine')
                    #silhouette_avg = silhouette_score(
                    #    torch.from_numpy(reduced_codes).cpu(), 
                    #    labels.cpu(), 
                    #    metric='cosine'
                    #)
                    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")
    
                    if silhouette_avg > best_silhouette:
                        best_labels = labels
                        best_silhouette = silhouette_avg
                        best_n_clusters = n_clusters
                        best_mus_sub = torch.stack([
                            KarcherMean(soft_assign=None, 
                                        codes=codes_k[torch.where(labels == i)[0].to(device=codes_k.device)],
                                        tol=1e-5, max_iter=100
                                       ).to(device=codes_k.device)
                            for i in range(n_clusters)
                        ])
                        counts = torch.tensor([
                            (labels == i).sum().item() for i in range(n_clusters)
                        ], dtype=torch.float32, device=codes.device)
                        data_covs_sub = compute_data_covs_hard_assignment(
                            labels.to(device), 
                            codes_k.to(device), 
                            n_clusters, 
                            best_mus_sub, 
                            prior
                        )
    
                        if use_priors:
                            best_covs_sub = []
                            D = codes.size(1)
                            for i in range(n_clusters):
                                if prior.prior_choice != 'dynamic_data_std':
                                    cov_k = prior.compute_post_cov(counts[i], data_covs_sub[i], D)
                                else:
                                    cov_k = prior.compute_post_cov(counts[i], data_covs_sub[i], D, k)
                                best_covs_sub.append(cov_k.to(codes.device))
                        else:
                            best_covs_sub = []
                            D = codes.size(1)
                            for i in range(n_clusters):
                                cov_k = data_covs_sub[i]
                                if not positive_definite.check(cov_k):
                                    print(f"[WARNING] Covariance matrix at cluster {i} is not positive definite. Adjusting...")
                                    cov_k = ensure_positive_definite(cov_k)
                                best_covs_sub.append(cov_k.to(codes.device))

    
                        best_pi_sub = counts / float(len(codes))
    
                        print(f"New best silhouette score: {best_silhouette} with {best_n_clusters} clusters.")
    
                if best_mus_sub is None:
                    print(f"No valid clustering found for cluster {k}. Initializing with default values.")
                    D = codes.size(1)
                    mus_sub = torch.zeros((n_sub, D), device=codes.device)
                    covs_sub = torch.stack([torch.eye(D, device=codes.device) for _ in range(n_sub)])
                    pi_sub = torch.zeros(n_sub, device=codes.device)
                    return mus_sub, covs_sub, pi_sub, best_n_clusters
                return best_mus_sub, best_covs_sub, best_pi_sub, best_n_clusters
    
    elif how_to_init_mu_sub in ["umap", "kmeans_1d"]:
        counts = []
        indices_k = logits.argmax(-1) == k
        codes_k = codes[indices_k]
        
        # If there are too few datapoints, create ghost subclusters.
        if len(codes_k) <= n_sub: 
            print('Empty or insufficient cluster:', codes_k.size())
            D = codes.size(1)
            mus_sub = torch.zeros((n_sub, D), device=codes.device)
            covs_sub = torch.stack([torch.eye(D, device=codes.device) for _ in range(n_sub)])
            pi_sub = torch.zeros(n_sub, device=codes.device)
            return mus_sub, covs_sub, pi_sub, n_sub
        
        else:
            # --------------------------
            # Dimensionality Reduction
            # --------------------------
            if how_to_init_mu_sub == "umap":
                n_neighbors = min(30, len(codes_k))
                init_method = 'random' if n_neighbors >= len(codes_k) else 'spectral'
                umap_obj = umap.UMAP(
                    init=init_method,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    n_components=3,
                    #random_state=random_state,
                    metric='cosine',
                ).fit(codes_k.detach().cpu())
                reduced_codes = umap_obj.embedding_
                print('UMAP SHAPE:', reduced_codes.shape)
            else:  # "kmeans_1d": use geodesic PCA for robustness to noise.
                # Compute a robust base point on the sphere using KarcherMean.
                mu0 = KarcherMean(soft_assign=None, codes=codes_k, tol=1e-5, max_iter=100)
                # Map codes to the tangent space at mu0 using the Riemannian logarithm.
                tangent_codes = Log_mapping(codes_k, mu0, standardization=False, normalization=False)
                # Apply PCA (with 1 component) on the tangent-space representation.
                pca_obj = PCA(n_components=1, random_state=random_state)
                reduced_codes = pca_obj.fit_transform(tangent_codes.detach().cpu().numpy())
                print('Geodesic PCA (tangent) SHAPE:', reduced_codes.shape)
            
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # --------------------------
            # Clustering & Subcluster Initialization
            # --------------------------
            if fixed_subclusters:
                # Fixed subclusters: directly cluster into n_sub groups.
                n_clusters = n_sub
                print(f"Fixed subclusters: Initializing with {n_clusters} subclusters.")
                if how_to_init_mu_sub == "kmeans_1d":
                    clustering_model = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=300, tol=1e-4)
                    clustering_model.fit(reduced_codes)
                    labels = torch.tensor(clustering_model.labels_, device=device)
                else:
                    clustering_model = PyTorchKMeans(
                        init='k-means++', 
                        max_iter=300, 
                        tol=1e-4, 
                        n_clusters=n_clusters, 
                        metric='euclidean',
                        #random_state=random_state,
                        verbose=False
                    )
                    labels = clustering_model.fit_predict(torch.from_numpy(reduced_codes).to(device=device))
    
                silhouette_avg = silhouette_score(
                    torch.from_numpy(reduced_codes).cpu(), 
                    labels.cpu(), 
                    metric='euclidean'
                )
                print(f"Silhouette Score for fixed {n_clusters} clusters: {silhouette_avg}")
    
                # Compute the subcluster means using the Karcher mean (over the original codes).
                mus_sub = torch.stack([
                    KarcherMean(soft_assign=None, 
                                codes=codes_k[torch.where(labels == i)[0].to(device=codes_k.device)],
                                tol=1e-5, max_iter=100
                               ).to(device=codes_k.device) 
                    for i in range(n_clusters)
                ])
                print(f'Initialized mus_sub shape: {mus_sub.shape}')
    
                # Compute covariance estimates with hard assignments.
                data_covs_sub = compute_data_covs_hard_assignment(
                    labels.to(device), 
                    codes_k.to(device), 
                    n_clusters, 
                    mus_sub, 
                    prior
                )
                counts = torch.tensor([
                    (labels == i).sum().item() for i in range(n_clusters)
                ], dtype=torch.float32, device=codes.device)
                if use_priors:
                    covs_sub = []
                    D = codes.size(1)
                    for i in range(n_clusters):
                        if prior.prior_choice != 'dynamic_data_std':
                            cov_k = prior.compute_post_cov(counts[i], data_covs_sub[i], D)
                        else:
                            cov_k = prior.compute_post_cov(counts[i], data_covs_sub[i], D, k)
                        covs_sub.append(cov_k.to(codes.device))
                else:
                    covs_sub = []
                    D = codes.size(1)
                    for i in range(n_clusters):
                        if not positive_definite.check(data_covs_sub[i]):
                          print('Covariance matrix not positive definite, adjusting...')
                          cov_k = ensure_positive_definite(data_covs_sub[i])
                        else :
                          cov_k=data_covs_sub[i]
                        covs_sub.append(cov_k.to(codes.device))
                    #covs_sub = data_covs_sub
    
                pi_sub = counts / float(len(codes))
    
                return mus_sub, covs_sub, pi_sub, n_clusters
            
            else:
                # Dynamic subcluster initialization: search over 2 to n_sub clusters.
                best_silhouette = -1
                best_mus_sub = None
                best_covs_sub = None
                best_pi_sub = None
                best_n_clusters = 2
                best_labels = None
    
                for n_clusters in range(2, n_sub + 1):
                    print(f"Trying {n_clusters} subclusters for cluster {k}.")
                    if how_to_init_mu_sub == "kmeans_1d":
                        clustering_model = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=300, tol=1e-4)
                        clustering_model.fit(reduced_codes)
                        labels = torch.tensor(clustering_model.labels_, device=device)
                    else:
                        clustering_model = PyTorchKMeans(
                            init='k-means++', 
                            max_iter=300, 
                            tol=1e-4, 
                            n_clusters=n_clusters, 
                            metric='euclidean',
                            random_state=random_state,
                            verbose=False
                        )
                        labels = clustering_model.fit_predict(torch.from_numpy(reduced_codes).to(device=device))
    
                    silhouette_avg = silhouette_score(
                        torch.from_numpy(reduced_codes).cpu(), 
                        labels.cpu(), 
                        metric='euclidean'
                    )
                    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")
    
                    if silhouette_avg > best_silhouette:
                        best_labels = labels
                        best_silhouette = silhouette_avg
                        best_n_clusters = n_clusters
                        best_mus_sub = torch.stack([
                            KarcherMean(soft_assign=None, 
                                        codes=codes_k[torch.where(labels == i)[0].to(device=codes_k.device)],
                                        tol=1e-5, max_iter=100
                                       ).to(device=codes_k.device)
                            for i in range(n_clusters)
                        ])
                        counts = torch.tensor([
                            (labels == i).sum().item() for i in range(n_clusters)
                        ], dtype=torch.float32, device=codes.device)
                        data_covs_sub = compute_data_covs_hard_assignment(
                            labels.to(device), 
                            codes_k.to(device), 
                            n_clusters, 
                            best_mus_sub, 
                            prior
                        )
    
                        if use_priors:
                            best_covs_sub = []
                            D = codes.size(1)
                            for i in range(n_clusters):
                                if prior.prior_choice != 'dynamic_data_std':
                                    cov_k = prior.compute_post_cov(counts[i], data_covs_sub[i], D)
                                else:
                                    cov_k = prior.compute_post_cov(counts[i], data_covs_sub[i], D, k)
                                best_covs_sub.append(cov_k.to(codes.device))
                        else:
                            best_covs_sub = []
                            D = codes.size(1)
                            for i in range(n_clusters):
                                if not positive_definite.check(data_covs_sub[i]):
                                  print('Covariance matrix not positive definite, adjusting...')
                                  cov_k = ensure_positive_definite(data_covs_sub[i])
                                else :
                                  cov_k=data_covs_sub[i]
                                best_covs_sub.append(cov_k.to(codes.device))
                            #best_covs_sub = data_covs_sub
    
                        best_pi_sub = counts / float(len(codes))
    
                        print(f"New best silhouette score: {best_silhouette} with {best_n_clusters} clusters.")
    
                if best_mus_sub is None:
                    print(f"No valid clustering found for cluster {k}. Initializing with default values.")
                    D = codes.size(1)
                    mus_sub = torch.zeros((n_sub, D), device=codes.device)
                    covs_sub = torch.stack([torch.eye(D, device=codes.device) for _ in range(n_sub)])
                    pi_sub = torch.zeros(n_sub, device=codes.device)
                    return mus_sub, covs_sub, pi_sub, best_n_clusters
                return best_mus_sub, best_covs_sub, best_pi_sub, best_n_clusters
    
    elif how_to_init_mu_sub == "soft_assign":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown how_to_init_mu_sub method: {how_to_init_mu_sub}")




def init_mus_and_covs_sub_v1(codes, k, mus, n_sub, how_to_init_mu_sub, logits, logits_sub, prior=None, use_priors=True, random_state=0, device="cpu"):
    print("On va initialiser mus_and_covs_sub")
    best_silhouette = -1
    best_mus_sub = None
    best_covs_sub = None
    best_pi_sub = None
    best_n_clusters = 2  # Start with the minimum number of clusters

    if how_to_init_mu_sub == "kmeans":
        pass
    elif how_to_init_mu_sub == "umap":
        # pca codes to 1D then perform 1d kmeans
        counts = []
        indices_k = logits.argmax(-1) == k
        codes_k = codes[indices_k]
        if len(codes_k) <= 2:
            # Handle empty cluster case
            print('empty cluster', codes_k.size())
            D = codes.size(1)  # Dimension of the data
            mus_sub = torch.zeros((n_sub, D), device=codes.device)  # Ghost mus set to zero or any neutral value
            covs_sub = torch.stack([torch.eye(D, device=codes.device) for _ in range(n_sub)])  # Identity matrices as ghost covs
            pi_sub = torch.zeros(n_sub, device=codes.device)  # Zero probabilities for ghost subclusters
            return mus_sub, covs_sub, pi_sub, best_n_clusters  # Return 2 as the default cluster number for empty case
        else:
            n_neighbors = min(30, len(codes_k))
            init_method = 'random' if n_neighbors >= len(codes_k) else 'spectral'    
            umap_obj = umap.UMAP(
                init=init_method,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                n_components=3,
                #random_state=42,
                metric='cosine'
            ).fit(codes_k.detach().cpu())
            umap_codes = umap_obj.embedding_

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # Loop over the range of subclusters from 2 to n_sub
            for n_clusters in range(2, n_sub + 1):
                kwargs = {
                    'metric': 'euclidean',
                    'distributed': False,
                    'random_state': 0,
                    'n_clusters': n_clusters,
                    'verbose': True
                }
                clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
                labels = clustering_model.fit_predict(torch.from_numpy(umap_codes).detach().to(device=device))
                cluster_centers = clustering_model.cluster_centers_

                # Compute silhouette score for current clustering
                silhouette_avg = silhouette_score(torch.from_numpy(umap_codes).detach().cpu(), labels.cpu(), metric='euclidean')
                #silhouette_avg = -1*davies_bouldin_score(
                #        torch.from_numpy(umap_codes).cpu().numpy(),
                #        labels.cpu().numpy()  # Convert labels to NumPy for compatibility
                #    )
                ##DAVIES BOULDIN
                #silhouette_avg=-1*davies_bouldin_score(torch.from_numpy(umap_codes).cpu(), labels.cpu())

                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_n_clusters = n_clusters  # Store the number of subclusters with the best score
                    best_mus_sub = torch.stack([KarcherMean(soft_assign=None, codes=codes_k[torch.where(labels==i)]).to(device=device) for i in range(n_clusters)])
                    counts = torch.tensor([len(codes_k[torch.where(labels==i)]) for i in range(n_clusters)])
                    data_covs_sub = compute_data_covs_hard_assignment(labels.to(device), codes_k.to(device), n_clusters, best_mus_sub, prior)

                    if use_priors:
                        best_covs_sub = []
                        D = len(codes_k[0])
                        for i in range(n_clusters):
                            if prior.prior_choice != 'dynamic_data_std':
                                covs_sub_k = prior.compute_post_cov(counts[i], data_covs_sub[i], D)
                            else:
                                covs_sub_k = prior.compute_post_cov(counts[i], data_covs_sub[i], D, k)
                            best_covs_sub.append(covs_sub_k)
                    else:
                        best_covs_sub = []
                        D = codes.size(1)
                        for i in range(n_clusters):
                            if not positive_definite.check(data_covs_sub[i]):
                              print('Covariance matrix not positive definite, adjusting...')
                              cov_k = ensure_positive_definite(data_covs_sub[i])
                            else :
                              cov_k=data_covs_sub[i]
                            best_covs_sub.append(cov_k.to(codes.device))
                        #best_covs_sub = data_covs_sub

                    best_pi_sub = counts / float(len(codes))

            # Return the best mus, covs, pi, and the number of clusters chosen
            return best_mus_sub, best_covs_sub, best_pi_sub, best_n_clusters
    elif how_to_init_mu_sub == "soft_assign":
        raise NotImplementedError()
        
def init_mus_and_covs_sub_2sub(codes,k, mus, n_sub, how_to_init_mu_sub, logits, logits_sub, prior=None, use_priors=True, random_state=0, device="cpu"):
    print("On va initialiser mus_and_covs_sub")
    if how_to_init_mu_sub == "kmeans":
        pass
    elif how_to_init_mu_sub == "umap":
        # pca codes to 1D then perform 1d kmeans
        counts = []
        indices_k = logits.argmax(-1) == k
        codes_k = codes[indices_k]
        if len(codes_k) <= n_sub:
            # empty cluster
            print('empty cluster',codes_k.size())
            #codes_k = codes
            #add the rest here 
            # Create ghost mus and covs
            D = codes.size(1)  # Dimension of the data
            mus_sub = torch.zeros((n_sub, D), device=codes.device)  # Ghost mus set to zero or any neutral value
            covs_sub = torch.stack([torch.eye(D, device=codes.device) for _ in range(n_sub)])  # Identity matrices as ghost covs
            pi_sub = torch.zeros(n_sub, device=codes.device)  # Zero probabilities for ghost subclusters
            return mus_sub, covs_sub, pi_sub
        # le else est une modif
        else:    
          print('MUS:',mus.size())
          print('k :',k)
          print('logits:',logits.size())
          print('codes_k :',codes_k.size())
          #proj_codes_k=Log_mapping(codes_k, mus[k])
          #pca = PCA(n_components=2).fit(proj_codes_k.detach().cpu())
          #pca_codes = pca.fit_transform(proj_codes_k.detach().cpu())
          n_neighbors = min(30, len(codes_k))
          init_method = 'random' if n_neighbors >= len(codes_k) else 'spectral'
          umap_obj = umap.UMAP(
                        n_neighbors=n_neighbors,
                        init=init_method,
                        min_dist=0.1,
                        n_components=3,
                        #random_state=42,
                        metric='cosine'
                    ).fit(codes_k.detach().cpu())
          umap_codes = umap_obj.embedding_
          print('UMAP SHAPE:',umap_codes.shape)
          #pca = PCA(n_components=1).fit(codes_k.detach().cpu())
          #pca_codes = pca.fit_transform(codes_k.detach().cpu())
          # kmeans = KMeans(n_clusters=n_sub, random_state=random_state).fit(pca_codes)
          device = "cuda:0" if torch.cuda.is_available() else "cpu"
          #labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes).to(device=device), num_clusters=n_sub, device=torch.device(device))
          #labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes), num_clusters=n_sub, device=torch.device("cpu"))
          kwargs = {
          'metric': 'euclidean',
          'distributed': False,
          #'random_state': 0,
          'n_clusters': n_sub,
          'verbose': True
      }
          clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
          labels=clustering_model.fit_predict(torch.from_numpy(umap_codes).detach().to(device=device))
          cluster_centers = clustering_model.cluster_centers_
  
          if len(codes[indices_k]) <= n_sub:
              c = torch.tensor([0, len(codes[indices_k])]) # sert a  ne pas prendre en compte ces subcluster car pi_sub =0 (dans le cas du sub loss egale KL_GMM?
              number, _ = torch.unique(labels, return_counts=True)
              #number=[0]
          else:
              number, c = torch.unique(labels, return_counts=True)
          counts.append(c)
          counts = counts[0]
  
          #mus_sub = torch.tensor(
          #    umap_obj.inverse_transform(cluster_centers.cpu().numpy()),
          #    device=device,
          #    requires_grad=False,
          #)
          mus_sub=torch.stack([KarcherMean(soft_assign=None, codes=codes_k[torch.where(labels==i)]).to(device=device) for i in number])
          print(f'LABELS {number} & count {counts}:')
          print('cluster_centers from kmeans :',cluster_centers.size())
          print('new_mus from KarcherMean: ', mus_sub.size())
          #new_mus=F.normalize(mus_sub.to(device),p=2,dim=1)
          #new_mus=Exp_mapping(mus_sub.to(device),mus[k].to(device))
          current_time = datetime.now().strftime("%H%M")
          torch.save(codes_k.detach().cpu(),f"/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/init_subcluster/into_mus_covs_sub_codes_{current_time}_{k}.pt")
          torch.save(labels,f"/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/init_subcluster/into_mus_covs_sub_labels_{current_time}_{k}.pt")
          #torch.save(mus_sub,f"/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/init_subcluster/into_mus_covs_sub_newmus_{current_time}_{k}.pt")
          print('SAVED init_mus_and_covs_sub')
          data_covs_sub = compute_data_covs_hard_assignment(labels.to(device), codes_k.to(device), n_sub, mus_sub, prior)
          
          if use_priors:
              #mus_sub = prior.compute_post_mus(counts, mus_sub.cpu())
              covs_sub = []
              D=len(codes_k[0])
              for i in range(n_sub):
                  if prior.prior_choice !='dynamic_data_std':
                     covs_sub_k = prior.compute_post_cov(counts[i], data_covs_sub[i],D)
                  else:
                     covs_sub_k = prior.compute_post_cov(counts[i], data_covs_sub[i],D,k)
                  covs_sub.append(covs_sub_k)
          else:
              covs_sub = []
              D=len(codes_k[0])
              for i in range(n_sub):
                  if not positive_definite.check(data_covs_sub[i]):
                    print('Covariance matrix not positive definite, adjusting...')
                    cov_k = ensure_positive_definite(data_covs_sub[i])
                  else :
                    cov_k=data_covs_sub[i]
                  covs_sub.append(cov_k.to(codes_k.device))
              #covs_sub = data_covs_sub
  
          pi_sub = counts / float(len(codes))
          #print("mus_sub type ",type(mus_sub))
          #print('mus_sub.size()',mus_sub[0].size())
        return mus_sub, covs_sub, pi_sub
        
    elif how_to_init_mu_sub == "soft_assign":
        raise NotImplementedError()
        
def init_mus_and_covs_sub_DPM(codes, k, n_sub, how_to_init_mu_sub, logits, logits_sub, prior=None, use_priors=True, random_state=0, device="cpu"):
    #print("On va initialiser mus_and_covs_sub")
    if how_to_init_mu_sub == "kmeans":
        #K.Cyril
        counts = []
        indices_k = logits.argmax(-1) == k
        codes_k = codes[indices_k]
        #print("logits.argmax(-1) :", logits.argmax(-1))
        #print('indices_k :', indices_k ,indices_k.size())
        if len(codes_k) <= n_sub:
            # empty cluster
            codes_k = codes
        
        #labels, cluster_centers = GPU_KMeans(X=codes_k.detach(), num_clusters=n_sub, device=torch.device('cuda:0'))
        labels, cluster_centers = GPU_KMeans(X=codes_k.detach(), num_clusters=n_sub, device=torch.device('cpu'))

        if len(codes[indices_k]) <= n_sub:
            c = torch.tensor([0, len(codes[indices_k])])
            counts= c #K.cyril
        else:
            _, c = torch.unique(labels, return_counts=True)
            counts = c #K.Cyril
        
        #print("cluster k :", k)
        #print("subclusters from k counts :" , counts)
        #counts.append(c)
        mus_sub = cluster_centers
        #print("X",X.size())
        #print("labels ",labels.size())
        #print("cluster ",cluster_centers)
        #print("COUNTS ",counts)

        data_covs_sub = compute_data_covs_hard_assignment(labels, codes_k, n_sub, mus_sub, prior)
        if use_priors:
            mus_sub = prior.compute_post_mus(counts, mus_sub.cpu())
            covs_sub = []
            for k in range(n_sub):
                covs_sub_k = prior.compute_post_cov(counts[k], codes_k[labels == k].mean(axis=0), data_covs_sub[k])
                covs_sub.append(covs_sub_k)
            covs_sub = torch.stack(covs_sub)
        else:
            
            covs_sub = data_covs_sub
        
        pi_sub = counts/float(len(codes))
        #pi_sub = torch.cat(counts) / float(len(codes))
        return mus_sub, covs_sub, pi_sub

    elif how_to_init_mu_sub == "kmeans_1d":
        # pca codes to 1D then perform 1d kmeans
        counts = []
        indices_k = logits.argmax(-1) == k
        codes_k = codes[indices_k]
        if len(codes_k) <= n_sub:
            # empty cluster
            codes_k = codes
        pca = PCA(n_components=1).fit(codes_k.detach().cpu())
        pca_codes = pca.fit_transform(codes_k.detach().cpu())
        # kmeans = KMeans(n_clusters=n_sub, random_state=random_state).fit(pca_codes)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes).to(device=device), num_clusters=n_sub, device=torch.device(device))
        labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes), num_clusters=n_sub, device=torch.device("cpu"))

        if len(codes[indices_k]) <= n_sub:
            c = torch.tensor([0, len(codes[indices_k])])
        else:
            _, c = torch.unique(labels, return_counts=True)
        counts.append(c)
        counts = counts[0]

        mus_sub = torch.tensor(
            pca.inverse_transform(cluster_centers.cpu().numpy()),
            device=device,
            requires_grad=False,
        )

        data_covs_sub = compute_data_covs_hard_assignment(labels, codes_k, n_sub, mus_sub, prior)
        if use_priors:
            mus_sub = prior.compute_post_mus(counts, mus_sub.cpu())
            covs_sub = []
            for k in range(n_sub):
                cov_sub_k = prior.compute_post_cov(counts[k], codes_k[labels == k].mean(axis=0), data_covs_sub[k])
                covs_sub.append(cov_sub_k)
            covs_sub = torch.stack(covs_sub)
        else:
            covs_sub = data_covs_sub

        pi_sub = counts / float(len(codes))
        return mus_sub, covs_sub, pi_sub

    elif how_to_init_mu_sub == "soft_assign":
        raise NotImplementedError()


def compute_data_sigma_sq_hard_assignment(labels, codes, K, mus):
    # returns K X D
    sigmas_sq = []
    for k in range(K):
        codes_k = codes[labels == k]
        sigmas_sq.append(codes_k.std(axis=0) ** 2)
    return torch.stack(sigmas_sq)


def KarcherMean(soft_assign=None, codes=None, cov=None, tol=1e-5, max_iter=100):
    """
    Compute the weighted Karcher mean on the unit sphere with Mahalanobis (whitening) weighting.

    Minimization Problem:
      min_{p in S^(D-1)} ?_i w_i * (log_p(x_i))^T * cov^{-1} * log_p(x_i)
    which can be rewritten by defining
      v'_i = S^(-1/2) * log_p(x_i)
    so that
      J(p) = ?_i w_i * ||v'_i||^2.
    The update is performed in the whitened tangent space and then mapped back.

    Args:
      soft_assign (torch.Tensor): Soft assignments (weights) for each data point (shape: [N]).
      codes (torch.Tensor): Data points (shape: [N, D]), expected to lie on the unit sphere.
      cov (torch.Tensor): Covariance matrix (shape: [D, D]) for Mahalanobis weighting.
      tol (float): Tolerance for convergence.
      max_iter (int): Maximum number of iterations.

    Returns:
      torch.Tensor: The computed Karcher mean on the unit sphere.
    """
    import torch

    # Use the same dtype and device as codes for all computations.
    dtype = codes.dtype
    device = codes.device
    
    # Normalize weights: if soft assignments provided, normalize them; otherwise, use uniform weights.
    if soft_assign is not None:
        soft_assign = soft_assign.to(dtype).to(device)
        weights = soft_assign / soft_assign.sum()
    else:
        weights = torch.ones(codes.size(0), device=device, dtype=dtype) / codes.size(0)

    # Normalize codes to ensure they lie on the unit sphere.
    codes_normalized = torch.nn.functional.normalize(codes, p=2, dim=-1)

    # Initial guess for p: weighted Euclidean mean, projected onto the sphere.
    p = torch.sum(weights.unsqueeze(-1) * codes_normalized, dim=0)
    p = torch.nn.functional.normalize(p, p=2, dim=-1)

    D = codes_normalized.size(1)
    # Pre-compute the whitening transformation:
    #cov=None
    if cov is not None:
        if not positive_definite.check(cov):
            print('Covariance matrix not positive definite, adjusting...')
            cov = ensure_positive_definite(cov)
        cov = cov.to(dtype).to(device)
        # Compute the Cholesky factor of cov: cov = L L^T (L lower-triangular)
        L = torch.linalg.cholesky(cov)
        # Compute the inverse of L.
        L_inv = torch.inverse(L)
        # Then, cov_sqrt_inv = L_inv^T satisfies cov_sqrt_inv @ cov_sqrt_inv = cov^{-1}
        cov_sqrt_inv = L_inv.T
    else:
        cov_sqrt_inv = torch.eye(D, device=device, dtype=dtype)

    # The inverse of cov_sqrt_inv is a valid square root of cov.
    cov_sqrt = torch.inverse(cov_sqrt_inv)

    for _ in range(max_iter):
        # Map each code to the tangent space at p.
        tangent_projections = Log_mapping(codes_normalized, p)  # shape: [N, D]

        # Whiten the tangent vectors: v'_i = cov_sqrt_inv * Log_p(x_i).
        tangent_projections_whitened = (cov_sqrt_inv @ tangent_projections.T).T

        # Compute the weighted average in the whitened tangent space.
        weighted_sum_whitened = torch.sum(weights.unsqueeze(-1) * tangent_projections_whitened, dim=0)

        # Transform back to the original tangent space:
        tangent_mean = cov_sqrt @ weighted_sum_whitened

        # Check convergence: if the norm of the tangent mean is small, we are near optimal.
        if torch.linalg.norm(tangent_mean, ord=2) < tol:
            break

        # Update p by mapping tangent_mean back onto the sphere using the exponential map.
        p = Exp_mapping(tangent_mean, p)
        p = torch.nn.functional.normalize(p, p=2, dim=-1)

    return p


def KarcherMean_previous(soft_assign=None, codes=None, tol=1e-5, max_iter=100):
    if soft_assign is not None:
      # Normalize soft_assign by summing
      weights = soft_assign / soft_assign.sum()
    else :
      weights=torch.ones(codes.size()[0])/codes.size()[0]
    
    # Normalize codes to ensure they lie on the unit sphere
    codes_normalized = F.normalize(codes, p=2, dim=-1)
    
    # Initial guess for p is the weighted mean of codes projected onto the sphere
    p = torch.sum(weights.unsqueeze(-1) * codes_normalized, dim=0)
    p = F.normalize(p, p=2, dim=-1)
    
    for _ in range(max_iter):
        # Project codes to the tangent space at p
        tangent_projections = Log_mapping(codes_normalized, p)
        
        # Compute the weighted mean in the tangent space
        tangent_mean = torch.sum(weights.unsqueeze(-1) * tangent_projections, dim=0, keepdim=True)
        
        # Check for convergence
        if torch.linalg.norm(tangent_mean, ord=2) < tol:
            break
        
        # Project the mean back onto the sphere to update p
        p = Exp_mapping(tangent_mean.squeeze(0), p)
        p = F.normalize(p, p=2, dim=-1)  # Ensure p stays on the sphere
        
    #print('INTO KARCHER MEAN :',p)
    return p

#True exp mapping to decoch
def Exp_mapping(codes, mu):
    # Ensure mu is normalized
    mu_normalized = F.normalize(mu, p=2, dim=-1).to(codes.device)
    
    # Compute the norm of each code vector in the tangent space
    codes_norm = torch.linalg.norm(codes, dim=-1, keepdim=True).to(codes.device)
    
    # Avoid division by zero by using clamp. We set a minimum value that is very small.
    safe_codes_norm = torch.clamp(codes_norm, min=1e-10)
    
    # Compute the components of the exponential map
    scaled_cos_component = mu_normalized * torch.cos(safe_codes_norm)
    scaled_sin_component = codes * torch.sin(safe_codes_norm) / safe_codes_norm
    
    # Sum the components to get points on the sphere
    x = scaled_cos_component + scaled_sin_component
    
    x=F.normalize(x,p=2,dim=-1)
    return x




def Exp_mapping_test(codes, mu, weights=None, alpha=1.0):
    """
    Applies a (weighted) Riemannian exponential mapping.

    Args:
        codes (torch.Tensor): Projected codes tensor of shape (N, D).
        mu (torch.Tensor): Mean tensor of shape (D,).
        weights (torch.Tensor, optional): Weighting tensor of shape (N,), values between 0 and 1.
            If provided, applies a weighted mapping where lower weights adjust the effective angle.
            Defaults to None.
        alpha (float, optional): Scaling factor for weight adjustment. Must match the alpha used in Log_mapping.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Reconstructed data tensor of shape (N, D).
    """
    if weights is not None:
        # Apply weighted mapping without tracking gradients
        with torch.no_grad():
            # Ensure mu is normalized
            mu_normalized = F.normalize(mu, p=2, dim=-1).to(codes.device)  # Shape: (D,)

            # Compute the norm of each code vector in the tangent space
            codes_norm = torch.linalg.norm(codes, dim=-1, keepdim=True).to(codes.device)  # Shape: (N, 1)

            # Avoid division by zero by using clamp. We set a minimum value that is very small.
            safe_codes_norm = torch.clamp(codes_norm, min=1e-10)  # Shape: (N, 1)

            # Compute theta_eff from the codes
            theta_eff = safe_codes_norm.squeeze(-1)  # Shape: (N,)

            # Recover the original theta using the weights and alpha
            theta = theta_eff / (1 + alpha * (1 - weights))  # Shape: (N,)

            # Clamp theta to [0, pi] to maintain valid angles
            theta = torch.clamp(theta, 0.0, math.pi)  # Shape: (N,)

            # Compute sin(theta)
            sin_theta = torch.sin(theta).unsqueeze(-1)  # Shape: (N, 1)
            sin_theta = torch.where(sin_theta == 0, torch.ones_like(sin_theta), sin_theta)  # Avoid division by zero

            # Reshape tensors for broadcasting
            mu_normalized = mu_normalized.unsqueeze(0)  # Shape: (1, D)
            theta = theta.unsqueeze(-1)  # Shape: (N, 1)

            # Compute the components of the exponential map
            scaled_cos_component = mu_normalized * torch.cos(theta)  # Shape: (N, D)
            scaled_sin_component = codes * torch.sin(theta) / sin_theta  # Shape: (N, D)

            # Sum the components to get points on the sphere
            x = scaled_cos_component + scaled_sin_component  # Shape: (N, D)

            # Normalize to ensure points lie on the unit sphere
            x = F.normalize(x, p=2, dim=-1)  # Shape: (N, D)

        return x
    else:
        # Apply standard mapping with gradient tracking
        # Ensure mu is normalized
        mu_normalized = F.normalize(mu, p=2, dim=-1).to(codes.device)  # Shape: (D,)

        # Compute the norm of each code vector in the tangent space
        codes_norm = torch.linalg.norm(codes, dim=-1, keepdim=True).to(codes.device)  # Shape: (N, 1)

        # Avoid division by zero by using clamp. We set a minimum value that is very small.
        safe_codes_norm = torch.clamp(codes_norm, min=1e-10)  # Shape: (N, 1)

        # Compute theta from the codes
        theta = safe_codes_norm.squeeze(-1)  # Shape: (N,)

        # Compute sin(theta)
        sin_theta = torch.sin(theta).unsqueeze(-1)  # Shape: (N, 1)
        sin_theta = torch.where(sin_theta == 0, torch.ones_like(sin_theta), sin_theta)  # Avoid division by zero

        # Reshape tensors for broadcasting
        mu_normalized = mu_normalized.unsqueeze(0)  # Shape: (1, D)
        theta = theta.unsqueeze(-1)  # Shape: (N, 1)

        # Compute the components of the exponential map
        scaled_cos_component = mu_normalized * torch.cos(theta)  # Shape: (N, D)
        scaled_sin_component = codes * torch.sin(theta) / sin_theta  # Shape: (N, D)

        # Sum the components to get points on the sphere
        x = scaled_cos_component + scaled_sin_component  # Shape: (N, D)

        # Normalize to ensure points lie on the unit sphere
        x = F.normalize(x, p=2, dim=-1)  # Shape: (N, D)

        return x

    
def Log_mapping_A_redecoche(codes, mu):
    # Normalize mu to ensure it lies on the unit sphere
    mu_normalized = F.normalize(mu, p=2, dim=-1).to(codes.device)
    
    # Compute cosine of theta using dot product
    dot_product = torch.sum(codes * mu_normalized, dim=-1)
    cos_theta = dot_product  # Dot product adjustment for individual pairs
    
    # Clamp cos_theta to [-1, 1] to avoid numerical issues with arccos
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # Compute theta
    theta = torch.acos(cos_theta)
    
    # Handle the theta = 0 case by setting sin(theta) = 1 when theta = 0
    sin_theta = torch.sin(theta)
    sin_theta[sin_theta == 0] = 1  # Avoid division by zero
    
    # Apply the Riemannian logarithm mapping formula
    # Adjustment for proper broadcasting and shape maintenance
    proj_codes = (codes - mu_normalized * cos_theta.unsqueeze(-1)) / sin_theta.unsqueeze(-1) * theta.unsqueeze(-1)
    
    return proj_codes


def Log_mapping_test(codes, mu, weights=None, alpha=0.5):
    """
    Applies a (weighted) Riemannian logarithm mapping.

    Args:
        codes (torch.Tensor): Input codes tensor of shape (N, D).
        mu (torch.Tensor): Mean tensor of shape (D,).
        weights (torch.Tensor, optional): Weighting tensor of shape (N,), values between 0 and 1.
            If provided, applies a weighted mapping where lower weights increase the effective angle.
            Defaults to None.
        alpha (float, optional): Scaling factor for weight adjustment. Controls the influence of weights.
            Higher values lead to greater separation for lower weights. Defaults to 1.0.

    Returns:
        torch.Tensor: Projected codes tensor of shape (N, D).
    """
    if weights is not None:
        # Apply weighted mapping without tracking gradients
        with torch.no_grad():
            # Normalize codes to ensure they lie on the unit sphere
            codes_normalized = F.normalize(codes, p=2, dim=-1)  # Shape: (N, D)

            # Normalize mu to ensure it lies on the unit sphere
            mu_normalized = F.normalize(mu, p=2, dim=-1)  # Shape: (D,)

            # Compute cosine of theta using dot product
            dot_product = torch.sum(codes * mu_normalized, dim=-1)  # Shape: (N,)
            cos_theta = torch.clamp(dot_product, -1.0, 1.0)  # Shape: (N,)

            # Compute theta
            theta = torch.acos(cos_theta)  # Shape: (N,)

            # Compute theta_eff
            theta_eff = theta * (1 + alpha * (1 - weights))  # Shape: (N,)

            # Clamp theta_eff to [0, pi] to maintain valid angles
            theta_eff = torch.clamp(theta_eff, 0.0, math.pi)  # Shape: (N,)

            # Compute sin(theta_eff)
            sin_theta_eff = torch.sin(theta_eff)  # Shape: (N,)
            sin_theta_eff = torch.where(sin_theta_eff == 0, torch.ones_like(sin_theta_eff), sin_theta_eff)  # Avoid division by zero

            # Reshape tensors for broadcasting
            mu_normalized = mu_normalized.unsqueeze(0)  # Shape: (1, D)
            cos_theta = cos_theta.unsqueeze(-1)  # Shape: (N, 1)
            theta_eff = theta_eff.unsqueeze(-1)  # Shape: (N, 1)
            sin_theta_eff = sin_theta_eff.unsqueeze(-1)  # Shape: (N, 1)

            # Apply the weighted Riemannian logarithm mapping formula
            proj_codes = (codes_normalized - mu_normalized * cos_theta) / sin_theta_eff * theta_eff  # Shape: (N, D)

        return proj_codes
    else:
        # Apply standard mapping with gradient tracking
        # Normalize codes to ensure they lie on the unit sphere
        codes_normalized = F.normalize(codes, p=2, dim=-1)  # Shape: (N, D)

        # Normalize mu to ensure it lies on the unit sphere
        mu_normalized = F.normalize(mu, p=2, dim=-1)  # Shape: (D,)

        # Compute cosine of theta using dot product
        dot_product = torch.sum(codes * mu_normalized, dim=-1)  # Shape: (N,)
        cos_theta = torch.clamp(dot_product, -1.0, 1.0)  # Shape: (N,)

        # Compute theta
        theta = torch.acos(cos_theta)  # Shape: (N,)

        # Compute sin(theta)
        sin_theta = torch.sin(theta)  # Shape: (N,)
        sin_theta = torch.where(sin_theta == 0, torch.ones_like(sin_theta), sin_theta)  # Avoid division by zero

        # Reshape tensors for broadcasting
        mu_normalized = mu_normalized.unsqueeze(0)  # Shape: (1, D)
        cos_theta = cos_theta.unsqueeze(-1)  # Shape: (N, 1)
        theta = theta.unsqueeze(-1)  # Shape: (N, 1)
        sin_theta = sin_theta.unsqueeze(-1)  # Shape: (N, 1)

        # Apply the Riemannian logarithm mapping formula
        proj_codes = (codes_normalized - mu_normalized * cos_theta) / sin_theta * theta  # Shape: (N, D)

        return proj_codes

def Log_mapping(codes, mu, standardization=False, normalization=False,scaling=False):
    """
    Performs the Riemannian logarithm mapping from the sphere to the tangent space,
    with optional standardization or normalization (min-max scaling) around a specified center (mu).
    
    Parameters:
    - codes (torch.Tensor): Input data points on the sphere (N x D).
    - mu (torch.Tensor): The center point (1 x D) for projection and scaling.
    - standardization (bool): If True, standardizes the projected codes.
    - normalization (bool): If True, applies min-max normalization to the projected data.
    
    Returns:
    - proj_codes (torch.Tensor): The projected (and optionally standardized or normalized) data (N x D).
    """
    # Normalize mu to ensure it lies on the unit sphere
    mu_normalized = F.normalize(mu, p=2, dim=-1).to(codes.device)
    
    # Compute cosine of theta using dot product
    dot_product = torch.sum(codes * mu_normalized, dim=-1)
    cos_theta = dot_product  # Dot product adjustment for individual pairs
    
    # Clamp cos_theta to [-1, 1] to avoid numerical issues with arccos
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # Compute theta
    theta = torch.acos(cos_theta)
    
    # Handle the theta = 0 case by setting sin(theta) = 1 when theta = 0
    sin_theta = torch.sin(theta)
    sin_theta[sin_theta == 0] = 1  # Avoid division by zero
    
    # Apply the Riemannian logarithm mapping formula
    # Adjustment for proper broadcasting and shape maintenance
    proj_codes = (codes - mu_normalized * cos_theta.unsqueeze(-1)) / sin_theta.unsqueeze(-1) * theta.unsqueeze(-1)
    
    # Optional standardization
    if standardization:
        proj_codes = standardize_data_with_center(proj_codes, mu_normalized)
    
    # Optional normalization (min-max scaling)
    if normalization:
        proj_codes = min_max_normalize(proj_codes)
    if scaling:
        proj_codes = 1.0*proj_codes
    return proj_codes


def standardize_data_with_center(data, center):
    """
    Standardizes the data around a specified center.
    
    Parameters:
    - data (torch.Tensor): The input data (N x D), where N is the number of points and D is the dimensionality.
    - center (torch.Tensor): The center point (1 x D) for standardization.
    
    Returns:
    - standardized_data (torch.Tensor): The standardized data (N x D).
    """
    # Ensure center is the same shape as data's feature dimension
    #center = center.view(1, -1)
    
    # Step 1: Shift data by subtracting the center
    shifted_data = data #- center
    
    # Step 2: Compute the standard deviation for each dimension
    std_dev = torch.std(shifted_data, dim=0, unbiased=False)
    
    # To prevent division by zero, ensure std_dev has no zeros
    std_dev = torch.where(std_dev == 0, torch.ones_like(std_dev), std_dev)
    
    # Step 3: Standardize data by dividing by the standard deviation
    standardized_data = shifted_data / std_dev
    
    return standardized_data


def min_max_normalize(data):
    """
    Applies min-max normalization to the data.
    
    Parameters:
    - data (torch.Tensor): The input data (N x D), where N is the number of points and D is the dimensionality.
    
    Returns:
    - normalized_data (torch.Tensor): The min-max normalized data (N x D).
    """
    # Step 1: Find the min and max values for each dimension
    data_min = torch.min(data, dim=0, keepdim=True)[0]
    data_max = torch.max(data, dim=0, keepdim=True)[0]
    
    # Step 2: Apply min-max normalization
    # To prevent division by zero, ensure no identical min and max values
    range_vals = data_max - data_min
    range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
    
    normalized_data = (data - data_min) / range_vals
    
    return normalized_data

def compute_data_covs_hard_assignment_v1(labels, codes, K, mus, prior):
      covs = []
      #print('labels into compute :',labels.shape)
      #print('codes into compute:',codes.shape)

      for k in range(K):
          codes_k = codes[labels == k]
          N_k = float(len(codes_k))
          if N_k > 0:
              codes_k=Log_mapping(codes_k, mus[k])
              cov_k = torch.matmul(
                  (codes_k).T,
                  (codes_k),
              )
              cov_k = cov_k / N_k
          else:
              if prior:
                  _, cov_k = prior.init_priors()
              else:
                  cov_k = torch.eye(codes.shape[1]) * 0.0005
          if not positive_definite.check(cov_k):
            print('INTO NOT POSTIVE DETECTION')
            cov_k.add_(0.00001 * torch.eye(cov_k.shape[1]))
          covs.append(cov_k)
      return torch.stack(covs)

import torch

import torch

def ensure_positive_definite(cov, initial_jitter=1e-5, max_iter=10, epsilon=1e-6):
    """
    Adjusts a covariance matrix to be positive definite.
    First attempts iterative jittering (with symmetry enforcement),
    then falls back to eigenvalue clipping if needed.
    
    Args:
        cov (torch.Tensor): The covariance matrix.
        initial_jitter (float): Starting jitter value for the diagonal.
        max_iter (int): Maximum iterations for jitter.
        epsilon (float): Minimum eigenvalue threshold for clipping.
        
    Returns:
        torch.Tensor: A positive definite covariance matrix.
        
    Raises:
        ValueError: If adjustments fail to produce a positive definite matrix.
    """
    # Ensure symmetry
    cov = (cov + cov.T) / 2.0
    
    jitter = initial_jitter
    I = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    
    # Attempt iterative jittering
    for i in range(max_iter):
        if positive_definite.check(cov):
            return cov
        else:
            cov = cov + jitter * I
            jitter *= 10  # Increase jitter exponentially
    
    # Jittering failed; fallback to eigenvalue clipping
    print("Jittering did not succeed. Falling back to eigenvalue clipping.")
    eigvals, eigvecs = torch.linalg.eigh(cov)
    print("Eigenvalues before clipping:", eigvals)
    
    eigvals_clipped = torch.clamp(eigvals, min=epsilon)
    cov_clipped = (eigvecs * eigvals_clipped) @ eigvecs.T
    # Enforce symmetry again after clipping
    cov_clipped = (cov_clipped + cov_clipped.T) / 2.0
    
    print("Eigenvalues after clipping:", torch.linalg.eigvalsh(cov_clipped))
    
    if positive_definite.check(cov_clipped):
        return cov_clipped
    else:
        raise ValueError("Covariance matrix could not be made positive definite after jittering and eigenvalue clipping.")


# Example usage within your covariance computation function:
def compute_data_covs_hard_assignment(labels, codes, K, mus, prior, log_scaling=False):
    covs = []
    
    print('labels shape:', labels.shape, 'device:', labels.device)
    print('codes shape:', codes.shape, 'device:', codes.device)
    print('mus shape:', mus.shape, 'device:', mus.device)
    if prior:
        print('prior exists')
    
    for k in range(K):
        codes_k = codes[labels == k]
        N_k = float(len(codes_k))
        
        print(f'codes_k shape (k={k}):', codes_k.shape, 'device:', codes_k.device)
        print(f'N_k (k={k}):', N_k)
        print('mus size :', mus.size())
        
        if N_k > 0:
            codes_k = Log_mapping(codes_k, mus[k].to(codes_k.device))
            print(f'codes_k after Log_mapping (k={k}):', codes_k.shape, 'device:', codes_k.device)
            
            cov_k = torch.matmul(codes_k.T, codes_k)
            print(f'cov_k shape after matmul (k={k}):', cov_k.shape, 'device:', cov_k.device)
            
            cov_k = cov_k / N_k
            print(f'cov_k after division by N_k (k={k}):', cov_k.shape, 'device:', cov_k.device)
        else:
            if prior:
                cov_k = prior.get_priors(k)
                print(f'cov_k from prior (k={k}):', cov_k.shape, 'device:', cov_k.device)
            else:
                cov_k = torch.eye(codes.shape[1], device=codes.device) * 0.0005
                print(f'cov_k as identity matrix (k={k}):', cov_k.shape, 'device:', cov_k.device)
        
        # Ensure the covariance matrix is positive definite
        if not positive_definite.check(cov_k):
            print('Covariance matrix not positive definite, adjusting...')
            cov_k = ensure_positive_definite(cov_k)
            print(f'cov_k after positive definite adjustment (k={k}):', cov_k.shape, 'device:', cov_k.device)
        
        covs.append(cov_k)
        print(f'Appended cov_k to covs (k={k}):', cov_k.shape, 'device:', cov_k.device)
    
    final_covs = torch.stack(covs)
    print('Final stacked covs shape:', final_covs.shape, 'device:', final_covs.device)
    return final_covs

    
def compute_data_covs_hard_assignment(labels, codes, K, mus, prior,log_scaling=False):
    covs = []
    
    # Print shapes and devices of input variables
    print('labels shape:', labels.shape, 'device:', labels.device)
    print('codes shape:', codes.shape, 'device:', codes.device)
    print('mus shape:', mus.shape, 'device:', mus.device)
    if prior:
        print('prior exists')
    
    for k in range(K):
        codes_k = codes[labels == k]
        N_k = float(len(codes_k))
        
        # Print intermediate variables' shapes and devices
        print(f'codes_k shape (k={k}):', codes_k.shape, 'device:', codes_k.device)
        print(f'N_k (k={k}):', N_k)
        print('mus size :',mus.size())
        
        if N_k > 0:
            codes_k = Log_mapping(codes_k, mus[k].to(codes_k.device))
            print(f'codes_k after Log_mapping (k={k}):', codes_k.shape, 'device:', codes_k.device)
            
            cov_k = torch.matmul(
                (codes_k).T,
                (codes_k),
            )
            print(f'cov_k shape after matmul (k={k}):', cov_k.shape, 'device:', cov_k.device)
            
            cov_k = cov_k / N_k
            print(f'cov_k after division by N_k (k={k}):', cov_k.shape, 'device:', cov_k.device)
        else:
            if prior:
                #_, cov_k = prior.init_priors()
                cov_k =prior.get_priors(k)
                print(f'cov_k from prior (k={k}):', cov_k.shape, 'device:', cov_k.device)
            else:
                cov_k = torch.eye(codes.shape[1]) * 0.0005
                print(f'cov_k as identity matrix (k={k}):', cov_k.shape, 'device:', cov_k.device)
        """
        if not positive_definite.check(cov_k):
            print('INTO NOT POSITIVE DETECTION')
            cov_k.add_(0.00001 * torch.eye(cov_k.shape[1],device=cov_k.device))
            print(f'cov_k after positive definite adjustment (k={k}):', cov_k.shape, 'device:', cov_k.device)
        """
        # Iteratively adjust cov_k to be positive definite if needed
        if not positive_definite.check(cov_k):
            print('Covariance matrix not positive definite, adjusting...')
            cov_k = ensure_positive_definite(cov_k)
            print(f'cov_k after positive definite adjustment (k={k}):', cov_k.shape, 'device:', cov_k.device)
        
        covs.append(cov_k)
        print(f'Appended cov_k to covs (k={k}):', cov_k.shape, 'device:', cov_k.device)
    
    final_covs = torch.stack(covs)
    print('Final stacked covs shape:', final_covs.shape, 'device:', final_covs.device)
    
    return final_covs

def compute_data_covs_hard_assignment_DPM(labels, codes, K, mus, prior):
    if prior and prior.mus_covs_prior.name == "NIG":
        return compute_data_sigma_sq_hard_assignment(labels, codes, K, mus)
    else:
        covs = []
        for k in range(K):
            codes_k = codes[labels == k]
            N_k = float(len(codes_k))
            if N_k > 0:
                cov_k = torch.matmul(
                    (codes_k - mus[k].cpu().repeat(len(codes_k), 1)).T,
                    (codes_k - mus[k].cpu().repeat(len(codes_k), 1)),
                )
                cov_k = cov_k / N_k
            else:
                if prior:
                    _, cov_k = prior.init_priors()
                else:
                    cov_k = torch.eye(codes.shape[1]) * 0.0005
            covs.append(cov_k)
        return torch.stack(covs)


def compute_data_sigma_sq_soft_assignment(codes, logits, K, mus):
    # Assuming the mus were also computed using soft assignments (mu is a weighted sample mean)

    denominator = logits.sum(axis=0)  # sum over all points per K
    stds = torch.stack([
        (logits[:, k].unsqueeze(1) * ((codes - mus[k])**2)).sum(axis=0) / denominator[k]
        for k in range(K)
    ])
    return stds

def compute_merged_mean_by_proportion(mus_list, N_i_list=None):
    """
    Merge clusters by running KarcherMean on the cluster-centers,
    weighted by their data proportions.
    
    mus_list : List[Tensor[D]]    the N cluster means
    N_i_list : List[float] or None the Ni for each cluster;
                                    if None, uses equal weights.
    """
    # Stack into (N, D)
    mus = torch.stack(mus_list, dim=0)
    N = mus.size(0)
    
    # Build weights
    if N_i_list is None or sum(N_i_list)==0:
        w = torch.ones(N, device=mus.device, dtype=mus.dtype) / N
    else:
        Ni = torch.tensor(N_i_list, device=mus.device, dtype=mus.dtype)
        w = Ni / Ni.sum()
    
    # Compute the (approximate) merged mean
    merged = KarcherMean(soft_assign=w, codes=mus, cov=None)
    return merged
    
def compute_merged_mean_karcher(codes_list):
    """
    Compute the merged cluster mean via Karcher mean on the hypersphere.
    
    Parameters:
    - codes_list: list of torch.Tensor of shape (Ni, D) for each cluster.
    
    Returns:
    - merged_mean: torch.Tensor of shape (D,), the merged cluster mean.
    """
    all_codes = torch.cat(codes_list, dim=0)
    merged_mean = KarcherMean(soft_assign=None, codes=all_codes, cov=None)
    return merged_mean

def compute_merged_mean(mus_list, N_i_list=None):
    """
    Computes the merged mean of N clusters on the unit sphere.

    Parameters:
    - mus_list: list of torch.Tensor of shape (D,), the cluster means.
    - N_i_list: list of floats, the number of points in each cluster.
                If None, equal weights are used.

    Returns:
    - merged_mean: torch.Tensor of shape (D,), the merged cluster mean.
    """
    n_clusters = len(mus_list)
    if N_i_list is None or sum(N_i_list) == 0:
        # Use equal weights
        weights = [1.0 / n_clusters] * n_clusters
    else:
        # Use provided N_i_list as weights
        total_weight = sum(N_i_list)
        weights = [Ni / total_weight for Ni in N_i_list]

    # Normalize the first mean
    v1 = mus_list[0]
    v1 = v1 / v1.norm()

    # Initialize lists for angles and weights
    angles = []
    total_weight = weights[0]

    # Compute angles between v1 and each vi (for i >= 2)
    for vi, wi in zip(mus_list[1:], weights[1:]):
        vi = vi / vi.norm()
        cos_theta = torch.dot(v1, vi)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)
        angles.append((theta, wi))
        total_weight += wi

    # Compute theta_star (optimal angle)
    numerator = sum(wi * theta for (theta, wi) in angles)
    denominator = total_weight
    theta_star = numerator / denominator if denominator != 0 else 0.0

    # Compute the weighted average of directions
    u_sum = torch.zeros_like(v1)
    for vi, wi in zip(mus_list[1:], weights[1:]):
        vi = vi / vi.norm()
        dot_product = torch.dot(v1, vi)
        if torch.abs(dot_product - 1.0) < 1e-6:
            # v1 and vi are the same; no contribution to direction
            continue
        elif torch.abs(dot_product + 1.0) < 1e-6:
            # v1 and vi are opposite; choose arbitrary orthogonal direction
            D = v1.size(0)
            idx = torch.argmin(torch.abs(v1))
            e = torch.zeros_like(v1)
            e[idx] = 1.0
            u = e - torch.dot(e, v1) * v1
            u = u / u.norm()
        else:
            # Compute the tangent vector from v1 to vi
            u = vi - dot_product * v1
            u = u / u.norm()
        u_sum += wi * u

    if u_sum.norm() < 1e-6:
        # All directions canceled out or no directions; merged mean is v1
        merged_mean = v1
    else:
        # Average direction
        u_avg = u_sum / u_sum.norm()
        # Compute merged mean by moving from v1 along u_avg by angle theta_star
        merged_mean = torch.cos(theta_star) * v1 + torch.sin(theta_star) * u_avg
        merged_mean = merged_mean / merged_mean.norm()
    return merged_mean



def rotation_matrix_from_a_to_b(a, b, percentage=1.0):
    assert b.size() == a.size()
    
    D_ = b.numel()
    dot = torch.dot(b.T, a)
    # Ensuring dot is within the bounds of [-1, 1]
    dot = torch.clamp(dot, -1.0, 1.0)
    
    if torch.abs(dot - 1.0) < 1e-6:
        bRa = torch.eye(D_)
    elif torch.abs(dot + 1.0) < 1e-6:
        bRa = -torch.eye(D_)
        bRa[0, 0] = torch.cos(percentage * torch.pi * 0.5)
        bRa[1, 1] = torch.cos(percentage * torch.pi * 0.5)
        bRa[0, 1] = -torch.sin(percentage * torch.pi * 0.5)
        bRa[1, 0] = torch.sin(percentage * torch.pi * 0.5)
    else:
        alpha = torch.acos(dot) * percentage
        
        c = a - b * dot
        assert torch.norm(c) > 1e-5
        c /= torch.norm(c)
        
        A = torch.outer(b, c) - torch.outer(c, b)
        temp = torch.outer(b, b) + torch.outer(c, c)
        temp2 = torch.cos(alpha) - 1
        
        bRa = torch.eye(D_) + torch.sin(alpha) * A + temp2 * temp
    
    return bRa

def rotate_vector_a_to_b(a, b, percentage=1.0):
    """
    Rotate vector `a` to align it with vector `b` using a specified rotation percentage.
    
    Parameters:
    - a (torch.Tensor): The initial vector.
    - b (torch.Tensor): The target vector to align with.
    - percentage (float): The percentage of the rotation to be applied.
    
    Returns:
    - torch.Tensor: The rotated vector `a` aligned towards vector `b`.
    """
    # Get the rotation matrix from `a` to `b`
    rotation_matrix = rotation_matrix_from_a_to_b(a, b, percentage)
    
    # Rotate vector `a` using the rotation matrix
    rotated_a = torch.matmul(rotation_matrix, a.unsqueeze(-1)).squeeze()
    
    return rotated_a

def compute_mus_soft_assignment(codes, logits, K,covs, constant=True):
    # Initialize mus tensor with the correct shape
    mus = torch.zeros((K, codes.shape[1]), device=codes.device, dtype=codes.dtype)
    
    for k in range(K):
        # Extract soft assignments for the k-th cluster
        soft_assign = logits[:, k]
        
        # Compute the Karcher mean for the k-th cluster
        mu_k = KarcherMean(soft_assign, codes,covs[k])
        
        # Assign the computed Karcher mean to the mus tensor
        mus[k, :] = mu_k
    
    if constant:
        mus = mus.detach()
    
    return mus

def compute_mus_soft_assignment_DPM(codes, logits, K, constant=True):
    # gives the embeddings (codes) and their probabilities to be sampled from the K classes, return each cluster's mu.
    # soft_assign (logits) is [N_batch X K], codes are [N_batch X feat_dim]
    denominator = logits.sum(axis=0)  # sum over all points per K
    # for each k, we are multiplying the k-th column of r with the codes matrix element-wise (first element * first row of c,...).
    # then, we are summing over all the data points (over the rows) and dividing by the normalizer
    # finally we are stacking all the mus.
    mus = torch.stack(
        [
            (logits[:, k].reshape(-1, 1) * codes).sum(axis=0) / denominator[k]
            for k in range(K)
        ]
    )  # K x feat_dim
    if constant:
        mus = mus.detach()
    return mus


def compute_pi_k(logits, prior=None):
    N = logits.shape[0]
    # sum for prob for each K (across all points) \sum_{i=1}^{N}P(z_i = k)
    r_sum = logits.sum(dim=0)
    if len(r_sum.shape) > 1:
        # this is sub clusters' pi need another sum
        r_sum = r_sum.sum(axis=0)
    pi = r_sum / torch.tensor(N, dtype=torch.float64)
    if prior:
        pi = prior.comp_post_pi(pi)
    return pi


def compute_data_covs_soft_assignment(logits, codes, K, mus, prior_name="NIW",log_scaling=False):
    # compute the data covs in soft assignment
    prior_name = prior_name or "NIW"
    if prior_name == "NIW":
        covs = []
        n_k = logits.sum(axis=0)
        n_k += 0.0001
        for k in range(K):
            if len(logits) == 0 or len(codes) == 0:
                # happens when finding subcovs of empty clusters
                cov_k = torch.eye(mus.shape[1]) * 0.0001
            else:
                proj_codes=Log_mapping(codes, mus[k])
                cov_k = torch.matmul(
                    (logits[:, k] * proj_codes.T),
                    (proj_codes),
                )
                cov_k = cov_k / n_k[k]
            """
            if not positive_definite.check(cov_k):
               cov_k.add_(0.00001 * torch.eye(cov_k.shape[1]))
            """
            if not positive_definite.check(cov_k):
              print('Covariance matrix not positive definite, adjusting...')
              cov_k = ensure_positive_definite(cov_k)
              print(f'cov_k after positive definite adjustment (k={k}):', cov_k.shape, 'device:', cov_k.device)
            covs.append(cov_k)
        return torch.stack(covs)
    elif prior_name == "NIG":
        return compute_data_sigma_sq_soft_assignment(logits=logits, codes=codes, K=K, mus=mus)

        
def compute_data_covs_soft_assignment_DPM(logits, codes, K, mus, prior_name="NIW"):
    # compute the data covs in soft assignment
    prior_name = prior_name or "NIW"
    if prior_name == "NIW":
        covs = []
        n_k = logits.sum(axis=0)
        n_k += 0.0001
        for k in range(K):
            if len(logits) == 0 or len(codes) == 0:
                # happens when finding subcovs of empty clusters
                cov_k = torch.eye(mus.shape[1]) * 0.0001
            else:
                cov_k = torch.matmul(
                    (logits[:, k] * (codes - mus[k].repeat(len(codes), 1)).T),
                    (codes - mus[k].repeat(len(codes), 1)),
                )
                cov_k = cov_k / n_k[k]
            covs.append(cov_k)
        return torch.stack(covs)
    elif prior_name == "NIG":
        return compute_data_sigma_sq_soft_assignment(logits=logits, codes=codes, K=K, mus=mus)


def compute_mus(codes, logits, pi, K, covs, how_to_compute_mu, use_priors=True, prior=None, random_state=0, device="cpu"):
    if how_to_compute_mu == "kmeans":
        kwargs = {
        'metric': 'cosine',
        'distributed': False,
        'random_state': 0,
        'n_clusters': K,
        'verbose': True
    }
        clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        labels=clustering_model.fit_predict(codes.detach())
        cluster_centers = clustering_model.cluster_centers_
        mus = cluster_centers
    elif how_to_compute_mu == "soft_assign":
        mus = compute_mus_soft_assignment(codes, logits, K,covs)

    #if use_priors:
    #    counts = pi * len(codes)
    #    mus = prior.compute_post_mus(counts, mus)
    #mus = mus
    return mus
    
def compute_mus_DPM(codes, logits, pi, K, how_to_compute_mu, use_priors=True, prior=None, random_state=0, device="cpu"):
    if how_to_compute_mu == "kmeans":
        labels, cluster_centers = GPU_KMeans(X=codes.detach(), num_clusters=K, device=torch.device('cpu'))
        mus = cluster_centers
    elif how_to_compute_mu == "soft_assign":
        mus = compute_mus_soft_assignment(codes, logits, K)

    if use_priors:
        counts = pi * len(codes)
        mus = prior.compute_post_mus(counts, mus)
    else:
        mus = mus
    return mus

def compute_covs(codes, logits, K, mus, use_priors=True, prior=None,log_scaling=False):
    from torch.distributions.constraints import positive_definite
    data_covs = compute_data_covs_soft_assignment(codes=codes, logits=logits, K=K, mus=mus, prior_name=prior.name if prior else None)
    covs = []
    r = logits.sum(axis=0)
    D=len(codes[0])
    for k in range(K):
        if use_priors:
            if prior.prior_choice=='dynamic_data_std':
              cov_k=prior.compute_post_cov(r[k],data_covs[k],D,psi_index=k)
            #cov_k = prior.compute_post_cov(r[k], mus[k], data_covs[k])
            else:
              cov_k=prior.compute_post_cov(r[k],data_covs[k],D)
            
        else :
            cov_k=data_covs[k]
        """
        if not positive_definite.check(cov_k):
            cov_k.add_(0.00001 * torch.eye(cov_k.shape[1]))
        """
        if not positive_definite.check(cov_k):
            print('Covariance matrix not positive definite, adjusting...')
            cov_k = ensure_positive_definite(cov_k)
            print(f'cov_k after positive definite adjustment (k={k}):', cov_k.shape, 'device:', cov_k.device)
        covs.append(cov_k)
    covs = torch.stack(covs)
    return covs



def compute_mus_covs_pis_subclusters(
    codes,
    logits,
    logits_sub,
    mus,         # main cluster means
    mus_sub,     # subcluster means
    covs_sub,    # subcluster covariances
    K,
    n_sub_list,
    hard_assignment=True,
    use_priors=True,
    prior=None
):
    """
    Computes updated subcluster mus, covs, and pi.
    Integrates your existing 'compute_data_covs_soft_assignment' method
    for computing multiple covariances at once.
    """

    # 1) Recompute subcluster mixture weights from logits_sub
    pi_sub = compute_pi_k(logits_sub, prior=prior if use_priors else None)
    print('INTO MUS COVS PIS SUBCLUSTER METHOD')

    how_to_init_mu_sub = "umap"  # or your chosen method
    mus_sub_new, covs_sub_new = [], []

    # Precompute where each cluster's subclusters start in logits_sub
    cluster_offsets = np.cumsum([0] + n_sub_list[:-1])

    for k in range(K):
        n_sub_k = n_sub_list[k]
        D = codes.size(1)

        # Hard assignment for main clusters
        if hard_assignment:
            indices = (logits.argmax(dim=-1) == k)
        else:
            # If not hard, might do something more complex
            indices = (logits.argmax(dim=-1) == k)

        codes_k = codes[indices]
        subcluster_offset = cluster_offsets[k]
        r_sub = logits_sub[indices, subcluster_offset : subcluster_offset + n_sub_k].to(codes.device)

        # Summation of responsibilities per subcluster
        denominator = r_sub.sum(dim=0)  # shape: (n_sub_k,)

        print(f"\nProcessing Main Cluster {k}")
        print(f" -> # points assigned: {indices.sum().item()}")
        print(f" -> # subclusters: {n_sub_k}, offset={subcluster_offset}")
        print(f" -> denominator (responsibilities): {denominator}")

        # Condition: reinit if too few total points in entire cluster
        # (You can add extra conditions: e.g. if only 1 subcluster is used, etc.)
        if indices.sum() < 2 or ( (denominator > 1e-5).sum() < 2 ) or (len(torch.unique(r_sub.argmax(dim=-1))) < 2):
            print(f"Cluster {k}: Reinitializing subclusters (degenerate: <2 points).")
            mus_sub_k, covs_sub_k, pi_sub_k, best_n_clusters = init_mus_and_covs_sub(
                codes=codes,
                k=k,
                mus=mus,    # main cluster means
                n_sub=n_sub_k,
                how_to_init_mu_sub=how_to_init_mu_sub,
                logits=logits,
                logits_sub=logits_sub,
                prior=prior,
                use_priors=use_priors,
                device=codes.device,
                fixed_subclusters=True
            )

            if best_n_clusters != n_sub_k:
                raise ValueError(
                    f"Mismatch in subcluster counts for cluster {k}: "
                    f"expected {n_sub_k}, got {best_n_clusters}"
                )

            # Append the newly initialized subcluster means/covs
            mus_sub_new.extend(mus_sub_k.to(codes.device))

            if how_to_init_mu_sub in ["kmeans_1d", "umap"]:
                covs_sub_new.extend([c.to(codes.device) for c in covs_sub_k])
            else:
                covs_sub_new.extend(covs_sub_k.to(codes.device))

            # Overwrite pi_sub portion
            pi_sub[subcluster_offset : subcluster_offset + best_n_clusters] = pi_sub_k.to(codes.device)

        else:
            # NORMAL CASE: We'll update subclusters that have data, 
            # but we do it in a single pass using compute_data_covs_soft_assignment.

            # 1) Compute new means for each subcluster
            new_subcluster_means = []
            old_mus_slice = mus_sub[subcluster_offset : subcluster_offset + n_sub_k]
            old_covs_slice = covs_sub[subcluster_offset : subcluster_offset + n_sub_k]

            for k_sub in range(n_sub_k):
                if denominator[k_sub] > 0:
                    # Weighted mean with responsibilities r_sub[:, k_sub]
                    z_sub = r_sub[:, k_sub]
                    mu_sub_k = KarcherMean(z_sub, codes_k,old_covs_slice[k_sub])
                else:
                    # Keep old mean if no points
                    mu_sub_k = old_mus_slice[k_sub]
                new_subcluster_means.append(mu_sub_k)

            # 2) Stack subcluster means into a single tensor to pass to compute_data_covs_soft_assignment
            mus_sub_k_tensor = torch.stack(new_subcluster_means)  # shape: [n_sub_k, D]

            # 3) Compute all subcluster covariances in one shot
            #    (Even if some subclusters have zero data, the function will handle them.)
            #    'K' in the function call is n_sub_k for subclusters, not the # main clusters
            data_covs_k = compute_data_covs_soft_assignment(
                logits=r_sub,           # shape: (num_points_in_cluster_k, n_sub_k)
                codes=codes_k,          # shape: (num_points_in_cluster_k, D)
                K=n_sub_k,              # we have n_sub_k subclusters
                mus=mus_sub_k_tensor,   # shape: (n_sub_k, D)
                prior_name=prior.name if use_priors else "NIW",
                log_scaling=False
            )  # shape: (n_sub_k, D, D)

            # 4) For each subcluster, decide if we keep the new cov or revert to old if it has zero points
            updated_mus_k = []
            updated_covs_k = []
            for k_sub in range(n_sub_k):
                if denominator[k_sub] > 0:
                    # Use newly computed mean, cov
                    mu_sub_k = mus_sub_k_tensor[k_sub]
                    cov_sub_k = data_covs_k[k_sub]

                    # If using a prior, do posterior update
                    if use_priors:
                        # Check for NaNs
                        if torch.isnan(cov_sub_k).any():
                            cov_sub_k = torch.eye(D, device=codes.device) * prior.mus_covs_prior.prior_sigma_scale

                        # Some prior logic
                        if prior.prior_choice == 'dynamic_data_std':
                            cov_sub_k = prior.compute_post_cov(denominator[k_sub], cov_sub_k, D, psi_index=k)
                        else:
                            cov_sub_k = prior.compute_post_cov(denominator[k_sub], cov_sub_k, D)

                    updated_mus_k.append(mu_sub_k)
                    updated_covs_k.append(cov_sub_k)

                else:
                    # No data -> keep old mean, old cov
                    updated_mus_k.append(old_mus_slice[k_sub])
                    updated_covs_k.append(old_covs_slice[k_sub])

            # 5) Add final subcluster params to the global lists
            mus_sub_new.extend(updated_mus_k)
            covs_sub_new.extend(updated_covs_k)

    # Convert lists to tensors
    mus_sub_new = torch.stack(mus_sub_new)      # shape: [sum(n_sub_list), D]
    covs_sub_new = torch.stack(covs_sub_new)    # shape: [sum(n_sub_list), D, D]

    print("\nFinal shapes:")
    print(f" -> mus_sub_new: {mus_sub_new.shape}")
    print(f" -> covs_sub_new: {covs_sub_new.shape}")
    print(f" -> pi_sub: {pi_sub.shape}")

    return mus_sub_new, covs_sub_new, pi_sub



def compute_mus_covs_pis_subclusters_v1(
    codes, logits, logits_sub, mus, mus_sub, covs_sub, K, n_sub_list, hard_assignment=True, use_priors=True, prior=None
):
    """
    Computes the mus (means), covs (covariances), and pi (weights) for subclusters,
    maintaining the same number of subclusters even in edge cases by creating ghost subclusters.
    
    Parameters:
    - n_sub_list: List of subcluster counts per main cluster (remains unchanged at the end).
    """
    pi_sub = compute_pi_k(logits_sub, prior=prior if use_priors else None)
    print('INTO MUS COVS PIS SUBCLUSTER METHOD')

    if hard_assignment:
        mus_sub_new, covs_sub_new = [], []
        subcluster_offset = 0  # Track subcluster index offsets

        for k in range(K):
            
        
            n_sub_k = n_sub_list[k]
            D = codes.size(1)  # Dimensionality of the codes

            indices = logits.argmax(-1) == k
            codes_k = codes[indices]
            r_sub = logits_sub[indices, subcluster_offset: subcluster_offset + n_sub_k].to(codes.device)
            denominator = r_sub.sum(axis=0)  # Sum over all points per subcluster
            # Get the current number of subclusters for cluster k
            print(f"Processing Cluster {k}")
            print(f"Number of Subclusters (n_sub_k): {n_sub_k}")
            print(f"Subcluster Offset: {subcluster_offset}")
        
            print(f"indices.sum(): {indices.sum().item()}")
            print(f"r_sub.shape: {r_sub.shape}")
            print(f"denominator: {denominator}")
            print(f"r_sub: {r_sub}")
            # Check for empty subclusters or insufficient data
            if indices.sum() < 2 or (denominator == 0).any() or len(torch.unique(r_sub.argmax(-1))) < n_sub_k:
                # Create ghost subclusters with zeroed mus, identity covariances, and zero pi
                print(f'Cluster {k}: Creating ghost subclusters due to insufficient data')

                mus_sub_ghost = torch.zeros((n_sub_k, D), device=codes.device)  # Ghost mus
                covs_sub_ghost = torch.stack([torch.eye(D, device=codes.device) for _ in range(n_sub_k)])  # Identity matrices as ghost covs
                pi_sub_ghost = torch.zeros(n_sub_k, device=codes.device)  # Zero probabilities for ghost subclusters

                # Append ghost subclusters to the output lists
                mus_sub_new.extend(mus_sub_ghost.to(codes.device))
                covs_sub_new.extend(covs_sub_ghost.to(codes.device))
                pi_sub[subcluster_offset: subcluster_offset + n_sub_k] = pi_sub_ghost.to(codes.device)

            else:
                # Otherwise, proceed with normal subcluster calculations
                mus_sub_k = []
                for k_sub in range(n_sub_k):
                    z_sub = r_sub[:, k_sub]
                    mus_sub_k.append(KarcherMean(z_sub, codes_k))

                mus_sub_new.extend(mus_sub_k)

                data_covs_k = compute_data_covs_soft_assignment(r_sub, codes_k, n_sub_k, mus_sub_k, prior.name)
                if use_priors:
                    covs_k = []
                    D = codes.size(1)
                    for k_sub in range(n_sub_k):
                        cov_k = data_covs_k[k_sub]
                        if torch.isnan(cov_k).any():
                            # Handle empty subclusters by creating an identity matrix
                            cov_k = torch.eye(cov_k.shape[0], device=codes.device) * prior.mus_covs_prior.prior_sigma_scale

                        if prior.prior_choice == 'dynamic_data_std':
                            cov_k = prior.compute_post_cov(r_sub.sum(axis=0)[k_sub], cov_k, D, psi_index=k)
                        else:
                            cov_k = prior.compute_post_cov(r_sub.sum(axis=0)[k_sub], cov_k, D)

                        covs_k.append(cov_k.to(codes.device))
                else:
                    covs_k = data_covs_k.to(codes.device)

                covs_sub_new.extend(covs_k)

            # Update subcluster offset
            subcluster_offset += n_sub_k
            

        mus_sub_new = torch.stack(mus_sub_new)
        covs_sub_new = torch.stack(covs_sub_new)

    return mus_sub_new, covs_sub_new, pi_sub


def compute_mus_covs_pis_subclusters_2sub(codes, logits, logits_sub,mus, mus_sub,covs_sub,K, n_sub, hard_assignment=True, use_priors=True, prior=None):
    pi_sub = compute_pi_k(logits_sub, prior=prior if use_priors else None)
    print('INTO MUS COVS PIS SUBCLUSTER METHODE')
    if hard_assignment:
        mus_sub_new, covs_sub_new = [], []
        for k in range(K):
            indices = logits.argmax(-1) == k
            codes_k = codes[indices]
            r_sub = logits_sub[indices, 2 * k: 2 * k + 2].to(codes.device)
            denominator = r_sub.sum(axis=0)  # sum over all points per K

            if indices.sum() < 2 or denominator[0] == 0 or denominator[1] == 0 or len(torch.unique(r_sub.argmax(-1))) < n_sub:
                # Empty subcluster encountered, re-initializing cluster {k}
                print('COMPUTE MUS COVS PIS SUBCLUSTER METHODE indices.sum <2 : NOTHING HAPPEN')
                #A decommenter
                mus_sub_n, covs_sub, pi_sub_ = init_mus_and_covs_sub(codes=codes, k=k,mus=mus,n_sub=n_sub, logits=logits, logits_sub=logits_sub, how_to_init_mu_sub="umap", prior=prior, use_priors=use_priors, device=codes.device)
                print('mus_sub_n :',mus_sub_n.size())
                pi_sub[2*k: 2*k+2] = pi_sub_
                mus_sub_new.append(mus_sub_n[0].to(codes.device))
                mus_sub_new.append(mus_sub_n[1].to(codes.device))
                covs_sub_new.append(covs_sub[0].to(codes.device))
                covs_sub_new.append(covs_sub[1].to(codes.device))
                #print('mus_sub_n[0]:',mus_sub_n[0].size())
                
                #print('mus_sub :',mus_sub.size())
                ##Modif 
                #mus_sub_=mus_sub[ 2 * k: 2 * k + 2]
                #print('mus_sub_',mus_sub_.size())
                #mus_sub_new.append(mus_sub_[0].to(codes.device))
                #mus_sub_new.append(mus_sub_[1].to(codes.device))
                #covs_sub_=covs_sub[ 2 * k: 2 * k + 2]
                #covs_sub_new.append(covs_sub_[0].to(codes.device))
                #covs_sub_new.append(covs_sub_[1].to(codes.device))
                
            else:
                mus_sub_k = []
                for k_sub in range(n_sub):
                    z_sub = r_sub[:, k_sub]
                    mus_sub= KarcherMean(z_sub,codes_k)
                    mus_sub_k.append(mus_sub)
                mus_sub_new.extend(mus_sub_k)
                data_covs_k = compute_data_covs_soft_assignment(r_sub, codes_k, n_sub, mus_sub_k, prior.name)
                if use_priors:
                    covs_k = []
                    D=len(codes[0])
                    for k_sub in range(n_sub):
                        cov_k = data_covs_k[k_sub]
                        if torch.isnan(cov_k).any():
                            # at least one of the subclusters has empty assignments
                            cov_k = torch.eye(cov_k.shape[0]) * prior.mus_covs_prior.prior_sigma_scale  # covs_sub[2 * k]
                        #cov_k = prior.compute_post_cov(r_sub.sum(axis=0)[k_sub], mus_sub_k[k_sub], cov_k)         
                        print("r_sub.sum(axis=0)[k_sub] size",r_sub.sum(axis=0)[k_sub].size())
                        if prior.prior_choice == 'dynamic_data_std':
                          cov_k = prior.compute_post_cov(r_sub.sum(axis=0)[k_sub], cov_k,D,psi_index=k)
                        else:
                          cov_k = prior.compute_post_cov(r_sub.sum(axis=0)[k_sub], cov_k,D)
                        
                        covs_k.append(cov_k.to(codes.device))
                else:
                    covs_k = data_covs_k.to(codes.device)
                covs_sub_new.extend(covs_k)
        mus_sub_new = torch.stack(mus_sub_new)
    #if use_priors:
    #    counts = pi_sub * len(codes)
    #    mus_sub_new = prior.compute_post_mus(counts, mus_sub_new)
    covs_sub_new = torch.stack(covs_sub_new)
    return mus_sub_new, covs_sub_new, pi_sub
  
  
def compute_mus_covs_pis_subclusters_DPM(codes, logits, logits_sub, mus_sub, K, n_sub, hard_assignment=True, use_priors=True, prior=None):
    pi_sub = compute_pi_k(logits_sub, prior=prior if use_priors else None)
    if hard_assignment:
        mus_sub_new, covs_sub_new = [], []
        for k in range(K):
            indices = logits.argmax(-1) == k
            codes_k = codes[indices]
            r_sub = logits_sub[indices, 2 * k: 2 * k + 2]
            denominator = r_sub.sum(axis=0)  # sum over all points per K

            if indices.sum() < 2 or denominator[0] == 0 or denominator[1] == 0 or len(torch.unique(r_sub.argmax(-1))) < n_sub:
                # Empty subcluster encountered, re-initializing cluster {k}
                mus_sub, covs_sub, pi_sub_ = init_mus_and_covs_sub(codes=codes, k=k, n_sub=n_sub, logits=logits, logits_sub=logits_sub, how_to_init_mu_sub="kmeans_1d", prior=prior, use_priors=use_priors, device=codes.device)
                pi_sub[2*k: 2*k+2] = pi_sub_
                mus_sub_new.append(mus_sub[0])
                mus_sub_new.append(mus_sub[1])
                covs_sub_new.append(covs_sub[0])
                covs_sub_new.append(covs_sub[1])
            else:
                mus_sub_k = []
                for k_sub in range(n_sub):
                    z_sub = r_sub[:, k_sub]
                    mus_sub_k.append(
                        (z_sub.reshape(-1, 1) * codes_k.cpu()).sum(axis=0)
                        / denominator[k_sub]
                    )
                mus_sub_new.extend(mus_sub_k)
                data_covs_k = compute_data_covs_soft_assignment(r_sub, codes_k, n_sub, mus_sub_k, prior.name)
                if use_priors:
                    covs_k = []
                    for k_sub in range(n_sub):
                        cov_k = data_covs_k[k_sub]
                        if torch.isnan(cov_k).any():
                            # at least one of the subclusters has empty assignments
                            cov_k = torch.eye(cov_k.shape[0]) * prior.mus_covs_prior.prior_sigma_scale  # covs_sub[2 * k]
                        cov_k = prior.compute_post_cov(r_sub.sum(axis=0)[k_sub], mus_sub_k[k_sub], cov_k)
                        covs_k.append(cov_k)
                else:
                    covs_k = data_covs_k
                covs_sub_new.extend(covs_k)
        mus_sub_new = torch.stack(mus_sub_new)
    if use_priors:
        counts = pi_sub * len(codes)
        mus_sub_new = prior.compute_post_mus(counts, mus_sub_new)
    covs_sub_new = torch.stack(covs_sub_new)
    return mus_sub_new, covs_sub_new, pi_sub


def cosine_dissimilarity_loss(output, target):
    normalized_output = nn.functional.normalize(output, dim=1)
    normalized_target = nn.functional.normalize(target, dim=1)
    loss = 1 - torch.sum(normalized_output * normalized_target, dim=1)
    return loss
        
        
def compute_mus_subclusters(codes, logits, logits_sub, pi_sub, mus_sub, K, n_sub, hard_assignment=True, use_priors=True, prior=None):
    if hard_assignment:
        # Data term
        mus_sub_new = []
        for k in range(K):
            denominator = logits_sub[:, 2 * k: 2 * k + 2].sum(
                    axis=0
                )  # sum over all points per K
            indices = logits.argmax(-1) == k
            if indices.sum() < 5:
                # empty cluster - do not change mu sub
                mus_sub_new.append(
                    mus_sub[2 * k: 2 * k + 2].clone().detach().cpu().type(torch.float32)
                )
            else:
                codes_k = codes[indices]
                for k_sub in range(n_sub):
                    if denominator[k_sub] == 0:
                        # empty cluster - do not change mu sub
                        mus_sub_new.append(
                            mus_sub[2 * k + k_sub].clone().detach().cpu().type(torch.float32).unsqueeze(0)
                        )
                    else:
                        z_sub = logits_sub[indices, 2 * k + k_sub]

                        mus_sub_new.append(
                            ((z_sub.reshape(-1, 1) * codes_k.cpu()).sum(axis=0)
                             / denominator[k_sub]).unsqueeze(0)
                        )
    mus_sub_new = torch.cat(mus_sub_new)

    if use_priors and prior:
        counts = pi_sub * len(codes)
        mus_sub_new = prior.compute_post_mus(counts, mus_sub_new)
    return mus_sub_new


def compute_covs_subclusters(codes, logits, logits_sub, K, n_sub, mus_sub, covs_sub, pi_sub, use_priors=True, prior=None):
    for k in range(K):
        indices = logits.argmax(-1) == k
        codes_k = codes[indices]
        r_sub = logits_sub[indices, 2 * k: 2 * k + 2]
        data_covs_k = compute_data_covs_soft_assignment(r_sub, codes_k, n_sub, mus_sub[2 * k: 2 * k + 2], prior.name)
        if use_priors:
            covs_k = []
            for k_sub in range(n_sub):
                cov_k = prior.compute_post_cov(r_sub.sum(axis=0)[k_sub], mus_sub[2 * k + k_sub], data_covs_k[k_sub])
                covs_k.append(cov_k)
            covs_k = torch.stack(covs_k)
        else:
            covs_k = data_covs_k
        if torch.isnan(cov_k).any():
            # at least one of the subclusters has empty assignments
            if torch.isnan(cov_k[0]).any():
                # first subcluster is empty give last cov
                covs_k[0] = covs_sub[2 * k]
            if torch.isnan(cov_k[1]).any():
                covs_k[1] = covs_sub[2 * k + 1]
        if k == 0:
            covs_sub_new = covs_k
        else:
            covs_sub_new = torch.cat([covs_sub_new, covs_k])
    return covs_sub_new

def map_k_sub_to_main_cluster_and_subcluster(k_sub, n_sub_list):
    """
    Maps a global subcluster index to its corresponding main cluster index and local subcluster index.

    Parameters:
    - k_sub: Global subcluster index (integer).
    - n_sub_list: List of integers, where each element represents the number of subclusters in the corresponding main cluster.

    Returns:
    - main_cluster_idx: Index of the main cluster to which k_sub belongs.
    - subcluster_local_idx: Local index of the subcluster within the main cluster.
    """
    cumulative_subclusters = np.cumsum([0] + n_sub_list)
    for idx in range(len(n_sub_list)):
        if cumulative_subclusters[idx] <= k_sub < cumulative_subclusters[idx + 1]:
            main_cluster_idx = idx
            subcluster_local_idx = k_sub - cumulative_subclusters[idx]
            return main_cluster_idx, subcluster_local_idx
    # If k_sub does not map to any cluster (should not happen), return last cluster
    return len(n_sub_list) - 1, n_sub_list[-1] - 1


    
def _create_subclusters(
    k_sub,
    codes,
    logits,
    logits_sub,
    mus_sub,
    pi_sub,
    n_sub_max,
    how_to_init_mu_sub,
    prior,
    n_sub_list,
    device=None,
    random_state=0,
    use_priors=True
):
    """
    Generalized _create_subclusters function for variable number of subclusters per cluster.

    Parameters:
    - k_sub: Global index of the subcluster to be split.
    - codes: Tensor of data points.
    - logits: Tensor of main cluster logits.
    - logits_sub: Tensor of subcluster logits.
    - mus_sub: Tensor of subcluster means.
    - pi_sub: Tensor of subcluster mixing coefficients.
    - n_sub_max: Maximum number of subclusters to create.
    - how_to_init_mu_sub: Method to initialize subcluster means ('umap', 'kmeans_1d', or 'kmeans').
    - prior: Prior object.
    - n_sub_list: List containing the number of subclusters per cluster.
    - device: Device to use ('cuda' or 'cpu').
    - random_state: Random seed for reproducibility.
    - use_priors: Whether to use prior distributions in calculations.

    Returns:
    - new_mus: List of new subcluster means.
    - new_covs: List of new subcluster covariances.
    - new_pis: List of new subcluster mixing coefficients.
    - best_n_subclusters: The number of subclusters selected.
    """
    device = device or codes.device
    D = mus_sub.shape[1]
    print("INTO CREATE_SUBCLUSTERS")

    if how_to_init_mu_sub == "kmeans":
        print("codes:", codes.device, codes.dtype, codes.shape)
        print("logits:", logits.device)
        print("logits_sub:", None if logits_sub is None else logits_sub.device)
        print("mus_sub:", mus_sub.device, mus_sub.shape)
        print("pi_sub:",  pi_sub.device,  pi_sub.shape)
        print("device arg:", device)

        # New branch: directly use raw codes for clustering.
        main_cluster_idx, subcluster_local_idx = map_k_sub_to_main_cluster_and_subcluster(k_sub, n_sub_list)
        indices_k = logits.argmax(-1) == main_cluster_idx
        codes_k = codes[indices_k]
        # Get indices of data points assigned to k_sub.
        if logits_sub is not None and len(logits_sub) > 0:
            subcluster_offset = sum(n_sub_list[:main_cluster_idx])
            sub_assignments = logits_sub.argmax(-1)
            codes_sub = codes[sub_assignments == k_sub]
        else:
            codes_sub = codes_k

        if len(codes_sub) <= 2:
            print('Empty or insufficient cluster:', codes_k.size())
            D = codes_sub.size(1)
            mus_sub = torch.zeros((2, D), device=codes_sub.device)
            covs_sub = torch.stack([torch.eye(D, device=codes_sub.device) for _ in range(2)])
            pi_sub = torch.zeros(2, device=codes_sub.device)
            best_n_subclusters=2
            return mus_sub, covs_sub, pi_sub, best_n_subclusters
            #return new_mus, new_covs, new_pis, best_n_subclusters
        else:
            # Use raw codes for clustering (no dimensionality reduction).
            reduced_codes = codes_sub.detach().cpu().numpy()
            print("Using raw codes for clustering. Shape:", reduced_codes.shape)

            # --------------------------
            # Clustering & Evaluation Loop
            # --------------------------
            best_silhouette = -1
            best_mus_sub = None
            best_covs_sub = None
            best_pi_sub = None
            best_n_subclusters = 2
            best_labels = None

            for n_clusters in range(2, n_sub_max + 1):
                clustering_model = PyTorchKMeans(
                    init='k-means++', 
                    max_iter=300, 
                    tol=1e-4, 
                    n_clusters=n_clusters, 
                    metric='cosine',
                    random_state=random_state,
                    verbose=False
                )
                labels = clustering_model.fit_predict(torch.from_numpy(reduced_codes).to(device=device))
                
                # Added check: Only compute silhouette_score if enough samples exist.
                if len(reduced_codes) <= n_clusters:
                    silhouette_avg = -1
                else:
                    silhouette_avg = _safe_silhouette(reduced_codes, labels.cpu(), metric='cosine')
                    #silhouette_avg = silhouette_score(reduced_codes, labels.cpu().numpy(), metric='cosine') #original
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_n_subclusters = n_clusters
                    best_labels = labels
                    best_mus_sub = [
                        KarcherMean(soft_assign=None, codes=codes_sub[labels == i], tol=1e-5, max_iter=100).to(device=device)
                        for i in range(n_clusters)
                    ]
                    counts = torch.tensor([torch.sum(labels == i).item() for i in range(n_clusters)], dtype=torch.float32)
                    data_covs_sub = compute_data_covs_hard_assignment(
                        labels, codes_sub, n_clusters, torch.stack(best_mus_sub), prior
                    )

                    if use_priors:
                        best_covs_sub = []
                        for i in range(n_clusters):
                            if prior.prior_choice != 'dynamic_data_std':
                                cov_sub_i = prior.compute_post_cov(counts[i], data_covs_sub[i], D)
                            else:
                                cov_sub_i = prior.compute_post_cov(counts[i], data_covs_sub[i], D, main_cluster_idx)
                            best_covs_sub.append(cov_sub_i)
                    else:
                        best_covs_sub = []
                        for i in range(n_clusters):
                            cov_sub_i = data_covs_sub[i]
                            if not positive_definite.check(cov_sub_i):
                                print(f"[WARNING] Covariance matrix at cluster {i} is not positive definite. Adjusting...")
                                cov_sub_i = ensure_positive_definite(cov_sub_i)
                            best_covs_sub.append(cov_sub_i.to(codes.device))


                    #best_pi_sub = counts / counts.sum() * pi_sub[k_sub]
                    # Make sure counts is on device
                    counts = torch.tensor(
                        [torch.sum(labels == i).item() for i in range(n_clusters)],
                        dtype=torch.float32,
                        device=codes.device,  # <-- add this
                    )
                    
                    # Align pi_sub with counts before multiplying
                    best_pi_sub = (counts / counts.sum()) * pi_sub[k_sub].to(codes.device)  # <-- add .to(device)

            # guard against degenerate "best" config (all points in one cluster)
            if (best_labels is None) or (torch.unique(best_labels).numel() == 1):
                print("[create_subclusters] best config collapsed to 1 cluster; using safe 2-cluster fallback.")
                D = codes_sub.size(1)
                mus_sub = torch.zeros((2, D), device=codes_sub.device)
                covs_sub = torch.stack([torch.eye(D, device=codes_sub.device) for _ in range(2)])
                pi_sub = torch.zeros(2, device=codes_sub.device)
                best_n_subclusters = 2
                return mus_sub, covs_sub, pi_sub, best_n_subclusters
  
            return best_mus_sub, best_covs_sub, best_pi_sub, best_n_subclusters

    elif how_to_init_mu_sub in ["umap", "kmeans_1d"]:
        # Map k_sub to main cluster index and local subcluster index.
        main_cluster_idx, subcluster_local_idx = map_k_sub_to_main_cluster_and_subcluster(k_sub, n_sub_list)
        indices_k = logits.argmax(-1) == main_cluster_idx
        codes_k = codes[indices_k]

        # Get indices of data points assigned to k_sub.
        if logits_sub is not None and len(logits_sub) > 0:
            subcluster_offset = sum(n_sub_list[:main_cluster_idx])
            sub_assignments = logits_sub.argmax(-1)
            codes_sub = codes[sub_assignments == k_sub]
        else:
            codes_sub = codes_k

        if len(codes_sub) <= 2:
            # Handle empty subcluster or insufficient data.
            print('Empty subcluster or insufficient data')
            """
            new_mus = [mus_sub[k_sub]]
            new_covs = [torch.eye(D, device=device)]
            new_pis = [pi_sub[k_sub]]
            best_n_subclusters = 1"""
            print('Empty or insufficient cluster:', codes_k.size())
            D = codes_sub.size(1)
            mus_sub = torch.zeros((2, D), device=codes_sub.device)
            covs_sub = torch.stack([torch.eye(D, device=codes_sub.device) for _ in range(2)])
            pi_sub = torch.zeros(2, device=codes_sub.device)
            best_n_subclusters=2
            return mus_sub, covs_sub, pi_sub, best_n_subclusters
            # (Optionally, you can implement a fallback assignment by minimum distance.)
            #raise ValueError('NUMBER OF ASSIGNED DATA THROUGH SUBCLUSTER NET INSUFFICIENT')
            #return new_mus, new_covs, new_pis, best_n_subclusters
        else:
            # --------------------------
            # Dimensionality Reduction
            # --------------------------
            if how_to_init_mu_sub == "umap":
                n_neighbors = min(30, len(codes_sub))
                init_method = 'random' if n_neighbors >= len(codes_sub) else 'spectral'
                umap_obj = umap.UMAP(
                    init=init_method,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    n_components=3,
                    random_state=random_state,
                    metric='cosine'
                ).fit(codes_sub.detach().cpu())
                reduced_codes = umap_obj.embedding_
                print("UMAP SHAPE:", reduced_codes.shape)
            else:  # how_to_init_mu_sub == "kmeans_1d"
                # Compute a robust base point using KarcherMean.
                mu0 = KarcherMean(soft_assign=None, codes=codes_sub, tol=1e-5, max_iter=100)
                # Map codes to the tangent space at mu0 using the logarithm map.
                tangent_codes = Log_mapping(codes_sub, mu0, standardization=False, normalization=False)
                # Apply PCA (with 1 component) on the tangent-space representation.
                pca_obj = PCA(n_components=1, random_state=random_state)
                reduced_codes = pca_obj.fit_transform(tangent_codes.detach().cpu().numpy())
                print("Geodesic PCA (tangent) SHAPE:", reduced_codes.shape)

            # --------------------------
            # Clustering & Evaluation Loop
            # --------------------------
            best_silhouette = -1
            best_mus_sub = None
            best_covs_sub = None
            best_pi_sub = None
            best_n_subclusters = 2  # Start from 2 subclusters
            best_labels = None

            for n_clusters in range(2, n_sub_max + 1):
                if how_to_init_mu_sub == "kmeans_1d":
                    clustering_model = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=300, tol=1e-4)
                    clustering_model.fit(reduced_codes)
                    labels = torch.tensor(clustering_model.labels_, device=device)
                else:
                    kwargs = {
                        'metric': 'euclidean',
                        'distributed': False,
                        'random_state': random_state,
                        'n_clusters': n_clusters,
                        'verbose': False
                    }
                    clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
                    labels = clustering_model.fit_predict(torch.from_numpy(reduced_codes).to(device=device))
                
                # Added check: Only compute silhouette_score if enough samples exist.
                if len(reduced_codes) <= n_clusters:
                    silhouette_avg = -1
                else:
                    silhouette_avg = silhouette_score(reduced_codes, labels.cpu().numpy(), metric='euclidean')

                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_n_subclusters = n_clusters
                    best_labels = labels
                    best_mus_sub = [
                        KarcherMean(soft_assign=None, codes=codes_sub[labels == i], tol=1e-5, max_iter=100).to(device=device)
                        for i in range(n_clusters)
                    ]
                    counts = torch.tensor([torch.sum(labels == i).item() for i in range(n_clusters)], dtype=torch.float32)
                    data_covs_sub = compute_data_covs_hard_assignment(
                        labels, codes_sub, n_clusters, torch.stack(best_mus_sub), prior
                    )

                    if use_priors:
                        best_covs_sub = []
                        for i in range(n_clusters):
                            if prior.prior_choice != 'dynamic_data_std':
                                cov_sub_i = prior.compute_post_cov(counts[i], data_covs_sub[i], D)
                            else:
                                cov_sub_i = prior.compute_post_cov(counts[i], data_covs_sub[i], D, main_cluster_idx)
                            best_covs_sub.append(cov_sub_i)
                    else:
                        best_covs_sub = []
                        for i in range(n_clusters):
                            cov_sub_i = data_covs_sub[i]
                            if not positive_definite.check(cov_sub_i):
                                print(f"[WARNING] Covariance matrix at sub-cluster {i} is not positive definite. Adjusting...")
                                cov_sub_i = ensure_positive_definite(cov_sub_i)
                            best_covs_sub.append(cov_sub_i.to(codes.device))

                    best_pi_sub = counts / counts.sum() * pi_sub[k_sub]

            return best_mus_sub, best_covs_sub, best_pi_sub, best_n_subclusters
    else:
        # Other options are not implemented.
        raise ValueError("Only 'umap', 'kmeans_1d', and 'kmeans' are implemented for how_to_init_mu_sub.")




def _create_subclusters_2sub(k_sub, codes, logits, logits_sub, mus_sub, pi_sub, n_sub, how_to_init_mu_sub, prior, device=None, random_state=0, use_priors=True):
    # k_sub is the index of sub mus that now turns into a mu
    # Recieves as input a vector of mus and generates two subclusters of it
    device= device or codes.device
    D = mus_sub.shape[1]
    print("INTO CREATE_SUBCLUSTER")
    if how_to_init_mu_sub == "soft_assign":
        mu_1 = (
            mus_sub[k_sub]
            + mus_sub[k_sub] @ torch.eye(D, D) * 0.05
        )
        mu_2 = (
            mus_sub[k_sub]
            - mus_sub[k_sub] @ torch.eye(D, D) * 0.05
        )
        new_covs = torch.stack([0.05 for i in range(2)])
        new_pis = torch.tensor([0.5, 0.5]) * pi_sub[k_sub]
        new_mus = torch.stack([mu_1, mu_2]).squeeze(dim=1)
        use_priors = False
        # return mus, covs, pis
    elif how_to_init_mu_sub == "kmeans" or "kmeans_1d" or "umap":
        indices_k = logits.argmax(-1) == int(k_sub / 2)
        codes_k = codes[indices_k, :]
        if len(logits_sub) > 0:
            sub_assignment = logits_sub.argmax(-1)
            codes_sub = codes[sub_assignment == k_sub]
        else:
            # comp assignments by min dist
            print('INTO ASSIGNMENTS BY MIN DIST')
            k_sub_other = k_sub + 1 if k_sub % 2 == 0 else k_sub - 1
            sub_assignment = comp_subclusters_params_min_dist(codes_k, mus_sub[k_sub], mus_sub[k_sub_other])
            codes_sub = codes_k[sub_assignment == (k_sub % 2)]  # sub_assignment is in range 0 and 1.

        if how_to_init_mu_sub == "kmeans":
            #labels, cluster_centers = GPU_KMeans(X=codes_sub.detach(), num_clusters=n_sub, device=torch.device('cuda:0'))
            kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 0,
            'n_clusters': n_sub,
            'verbose': True }
            clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
            labels=clustering_model.fit_predict(codes_sub.detach())
            cluster_centers = clustering_model.cluster_centers_
            #labels, cluster_centers = GPU_KMeans(X=codes_sub.detach(), num_clusters=n_sub, device=torch.device('cpu'))
            new_mus = cluster_centers.cpu()
            new_covs = compute_data_covs_hard_assignment(labels=labels, codes=codes_sub, K=n_sub, mus=new_mus, prior=prior)
            _, new_pis = torch.unique(
                labels, return_counts=True
            )
            new_pis = (new_pis / float(len(codes_sub))) * pi_sub[k_sub]
        
        elif how_to_init_mu_sub == "kmeans_1d":
            # kmeans_1d
            proj_codes_sub=Log_mapping(codes_sub, mus_sub[k_sub])
            pca = PCA(n_components=3).fit(proj_codes_sub.detach().cpu())
            pca_codes = pca.fit_transform(proj_codes_sub.detach().cpu())
            
            #labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes).to(device=device), num_clusters=n_sub, device=torch.device(device))
            #labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes), num_clusters=n_sub, device=torch.device("cpu"))
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
            kwargs = {
            'metric': 'euclidean',
            'distributed': False,
            'random_state': 0,
            'n_clusters': n_sub,
            'verbose': True
        }
            clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
            labels=clustering_model.fit_predict(torch.from_numpy(pca_codes).detach().to(device=device))
            cluster_centers = clustering_model.cluster_centers_


            new_mus = torch.tensor(
                pca.inverse_transform(cluster_centers.cpu().numpy()),
                device=device,
                requires_grad=False,
            )
            new_mus=Exp_mapping(new_mus,mus_sub[k_sub]).cpu()
            
            new_covs = compute_data_covs_hard_assignment(
                labels=labels, codes=codes_sub, K=n_sub, mus=new_mus, prior=prior
            )
            _, new_pis = torch.unique(
                labels, return_counts=True
            )
            new_pis = (new_pis.to(pi_sub.device) / float(len(codes_sub))) * pi_sub[k_sub]
            
        

        elif how_to_init_mu_sub == "umap":
                # kmeans_1d
                #proj_codes_sub=Log_mapping(codes_sub, mus_sub[k_sub])
                #pca = PCA(n_components=1).fit(proj_codes_sub.detach().cpu())
                #pca_codes = pca.fit_transform(proj_codes_sub.detach().cpu())
                n_neighbors = min(30, len(codes_sub))
                init_method = 'random' if n_neighbors >= len(codes_sub) else 'spectral'
                umap_obj = umap.UMAP(
                      init=init_method,
                      n_neighbors=n_neighbors,
                      min_dist=0.1,
                      n_components=3,
                      random_state=42,
                      metric='cosine'
                  ).fit(codes_sub.detach().cpu())
                umap_codes = umap_obj.embedding_
                print('UMAP SHAPE createsubcluster:',umap_codes.shape)
                
                #labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes).to(device=device), num_clusters=n_sub, device=torch.device(device))
                #labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes), num_clusters=n_sub, device=torch.device("cpu"))
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
                kwargs = {
                'metric': 'euclidean',
                'distributed': False,
                'random_state': 0,
                'n_clusters': n_sub,
                'verbose': True
            }
                clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
                labels=clustering_model.fit_predict(torch.from_numpy(umap_codes).detach().to(device=device))
                cluster_centers = clustering_model.cluster_centers_
    
                
                #new_mus = torch.tensor(
                #    umap_obj.inverse_transform(cluster_centers.cpu().numpy()),
                #    device=device,
                #    requires_grad=False,
                #)
                #new_mus=F.normalize(new_mus,p=2,dim=1).cpu()
                number, new_pis = torch.unique(
                    labels, return_counts=True
                )
                new_mus=torch.stack([KarcherMean(soft_assign=None, codes=codes_sub[torch.where(labels==i)]).cpu() for i in number]).cpu()
                print(f'LABELS {number} & count {new_pis}')
                print('cluster_centers from kmeans :',cluster_centers.size())
                print('new_mus from KarcherMean: ', new_mus.size())
                
                current_time = datetime.now().strftime("%H%M")
                #new_mus=Exp_mapping(new_mus,mus_sub[k_sub]).cpu()
                torch.save(codes_sub.detach().cpu(),f"/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/init_subcluster/create_subcluster_fct/into_create_subclusters_codes_{current_time}.pt")
                torch.save(labels,f"/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/init_subcluster/create_subcluster_fct/into_create_subclusters_labels_{current_time}.pt")
                torch.save(new_mus,f"/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/init_subcluster/create_subcluster_fct/into_create_subclusters_newmus_{current_time}.pt")
                
                
                new_covs = compute_data_covs_hard_assignment(
                    labels=labels, codes=codes_sub, K=n_sub, mus=new_mus, prior=prior
                )
                new_pis = (new_pis.to(pi_sub.device) / float(len(codes_sub))) * pi_sub[k_sub]
                
            
        if use_priors:
            _, counts = torch.unique(labels, return_counts=True)
            #new_mus = prior.compute_post_mus(counts, new_mus)  # up until now we didn't use this
            covs = []
            D=len(codes[0])
            for k in range(n_sub):
                if prior.prior_choice !='dynamic_data_std':
                  new_cov_k = prior.compute_post_cov(counts[k],new_covs[k],D)
                else:
                  new_cov_k = prior.compute_post_cov(counts[k],new_covs[k],D,int(k_sub / 2))
                covs.append(new_cov_k)
            new_covs = torch.stack(covs)
            pis_post = prior.comp_post_pi(new_pis)  # sum to 1
            new_pis = pis_post * pi_sub[k_sub]  # sum to pi_sub[k_sub]
        return new_mus, new_covs, new_pis

def _create_subclusters_DPM(k_sub, codes, logits, logits_sub, mus_sub, pi_sub, n_sub, how_to_init_mu_sub, prior, device=None, random_state=0, use_priors=True):
    # k_sub is the index of sub mus that now turns into a mu
    # Recieves as input a vector of mus and generates two subclusters of it
    device= device or codes.device
    D = mus_sub.shape[1]
    print("INIT SUBCLUSTER")
    if how_to_init_mu_sub == "soft_assign":
        mu_1 = (
            mus_sub[k_sub]
            + mus_sub[k_sub] @ torch.eye(D, D) * 0.05
        )
        mu_2 = (
            mus_sub[k_sub]
            - mus_sub[k_sub] @ torch.eye(D, D) * 0.05
        )
        new_covs = torch.stack([0.05 for i in range(2)])
        new_pis = torch.tensor([0.5, 0.5]) * pi_sub[k_sub]
        new_mus = torch.stack([mu_1, mu_2]).squeeze(dim=1)
        use_priors = False
        # return mus, covs, pis
    elif how_to_init_mu_sub == "kmeans" or "kmeans_1d":
        indices_k = logits.argmax(-1) == int(k_sub / 2)
        codes_k = codes[indices_k, :]
        
        if len(logits_sub) > 0:
            sub_assignment = logits_sub.argmax(-1)
            codes_sub = codes[sub_assignment == k_sub]
        else:
            # comp assignments by min dist
            k_sub_other = k_sub + 1 if k_sub % 2 == 0 else k_sub - 1
            sub_assignment = comp_subclusters_params_min_dist(codes_k, mus_sub[k_sub], mus_sub[k_sub_other])
            codes_sub = codes_k[sub_assignment == (k_sub % 2)]  # sub_assignment is in range 0 and 1.

        if how_to_init_mu_sub == "kmeans":
            #labels, cluster_centers = GPU_KMeans(X=codes_sub.detach(), num_clusters=n_sub, device=torch.device('cuda:0'))
            labels, cluster_centers = GPU_KMeans(X=codes_sub.detach(), num_clusters=n_sub, device=torch.device('cpu'))
            new_mus = cluster_centers.cpu()
            print('labels into compute :',labels.shape)
            print('codes into compute:',codes_sub.shape)
            new_covs = compute_data_covs_hard_assignment(labels=labels, codes=codes_sub, K=n_sub, mus=new_mus, prior=prior)
            _, new_pis = torch.unique(
                labels, return_counts=True
            )
            new_pis = (new_pis / float(len(codes_sub))) * pi_sub[k_sub]
        elif how_to_init_mu_sub == "kmeans_1d":
            # kmeans_1d
            pca = PCA(n_components=3).fit(codes_sub.detach().cpu())
            pca_codes = pca.fit_transform(codes_sub.detach().cpu())

            device = "cuda" if torch.cuda.is_available() else "cpu"
            #labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes).to(device=device), num_clusters=n_sub, device=torch.device(device))
            labels, cluster_centers = GPU_KMeans(X=torch.from_numpy(pca_codes), num_clusters=n_sub, device=torch.device("cpu"))

            new_mus = torch.tensor(
                pca.inverse_transform(cluster_centers.cpu().numpy()),
                device=device,
                requires_grad=False,
            ).cpu()
            new_covs = compute_data_covs_hard_assignment(
                labels=labels, codes=codes_sub, K=n_sub, mus=new_mus, prior=prior
            )
            _, new_pis = torch.unique(
                labels, return_counts=True
            )
            new_pis = (new_pis / float(len(codes_sub))) * pi_sub[k_sub]

        if use_priors:
            _, counts = torch.unique(labels, return_counts=True)
            new_mus = prior.compute_post_mus(counts, new_mus)  # up until now we didn't use this
            covs = []
            for k in range(n_sub):
                new_cov_k = prior.compute_post_cov(counts[k], codes_sub[labels == k].mean(axis=0), new_covs[k])
                covs.append(new_cov_k)
            new_covs = torch.stack(covs)
            pis_post = prior.comp_post_pi(new_pis)  # sum to 1
            new_pis = pis_post * pi_sub[k_sub]  # sum to pi_sub[k_sub]

    return new_mus, new_covs, new_pis

def comp_subclusters_params_min_dist(codes_k, mu_sub_1, mu_sub_2):
    """
    Comp assignments to subclusters by max cosine similarity to subclusters centers.
    codes_k (torch.tensor): the datapoints assigned to the k-th cluster
    mu_sub_1, mu_sub_2 (torch.tensor, torch.tensor): the centroids of the first and second subclusters of cluster k

    Returns the (hard) assignments vector (in range 0 and 1).
    Can be used for e.g.,
    codes_k_1 = codes_k[assignments == 0]
    codes_k_2 = codes_k[assignments == 1]
    """

    # Normalize centroids to have unit norm for cosine similarity computation
    mu_sub_1_norm = F.normalize(mu_sub_1, p=2, dim=1)
    mu_sub_2_norm = F.normalize(mu_sub_2, p=2, dim=1)

    # Calculate cosine similarity for each datapoint to each centroid
    # Note: expand_as ensures that the centroids are broadcasted correctly to match the size of codes_k
    sim_0 = F.cosine_similarity(codes_k, mu_sub_1_norm.unsqueeze(0).expand_as(codes_k), dim=1)
    sim_1 = F.cosine_similarity(codes_k, mu_sub_2_norm.unsqueeze(0).expand_as(codes_k), dim=1)

    # Assign datapoints to the closest centroid based on the higher cosine similarity
    assignments = torch.stack([sim_0, sim_1]).argmax(0)
    return assignments


def comp_subclusters_params_min_dist_DPM(codes_k, mu_sub_1, mu_sub_2):
    """
    Comp assignments to subclusters by min dist to subclusters centers
    codes_k (torch.tensor): the datapoints assigned to the k-th cluster
    mu_sub_1, mu_sub_2 (torch.tensor, torch.tensor): the centroids of the first and second subclusters of cluster k

    Returns the (hard) assignments vector (in range 0 and 1).
    can be used for e.g.,
    codes_k_1 = codes_k[assignments == 0]
    codes_k_2 = codes_k[assignments == 1]
    """

    dists_0 = torch.sqrt(torch.sum((codes_k - mu_sub_1) ** 2, axis=1))
    dists_1 = torch.sqrt(torch.sum((codes_k - mu_sub_2) ** 2, axis=1))
    assignments = torch.stack([dists_0, dists_1]).argmin(0)
    return assignments
