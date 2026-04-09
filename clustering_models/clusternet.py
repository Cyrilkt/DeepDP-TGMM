# fast_clusternet.py
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen

import numpy as np
import torch
import torch.nn as nn
from src.clustering_models.clusternet_modules.clusternet_trainer import (
    ClusterNetTrainer,
)
from src.clustering_models.clusternet_modules.utils.clustering_utils.clustering_operations import compute_merged_mean

def cosine_dissimilarity_batch(X, clusters, epsilon=1e-10):
    """
    Compute the cosine dissimilarity matrix D between rows of X and rows of clusters:
        D_ij = 1 - (X_i · clusters_j) / (||X_i|| * ||clusters_j||).
    Vectorized, BLAS‐backed, no Python loops.
    """
    # 1) normalize each row to unit ℓ2‐norm
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=epsilon)
    C_norm = clusters / np.linalg.norm(clusters, axis=1, keepdims=True).clip(min=epsilon)
    # 2) fast dot‐product
    sim_mat = X_norm @ C_norm.T       # shape (n_samples, n_clusters)
    # 3) return dissimilarity
    return 1.0 - sim_mat

class ClusterNet(object):
    def __init__(self, args, feature_extractor):
        self.args = args
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters
        self.clusters = np.zeros((self.n_clusters, self.latent_dim))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = args.n_jobs  # no longer used for distance
        self.feature_extractor = feature_extractor
        self.device = f"cuda:{args.gpus}" if torch.cuda.is_available() and args.gpus is not None else "cpu"

    def _compute_dist(self, X):
        """
        X: numpy array of shape (n_samples, latent_dim)
        returns: numpy array (n_samples, n_clusters)
        """
        return cosine_dissimilarity_batch(X, self.clusters)

    def init_cluster(self, train_loader, val_loader, logger, indices=None, centers=None, init_num=0):
        """ Generate initial clusters using the clusternet """
        self.feature_extractor.freeze()
        self.model = ClusterNetTrainer(
            args=self.args,
            init_k=self.n_clusters,
            latent_dim=self.latent_dim,
            feature_extractor=self.feature_extractor,
            centers=centers,
            init_num=init_num
        )
        self.fit_cluster(train_loader, val_loader, logger, centers)
        self.model.cluster_model.freeze()
        self.feature_extractor.unfreeze()
        self.feature_extractor.to(device=self.device)

    def fit_cluster(self, train_loader, val_loader, logger, centers=None):
        self.feature_extractor.freeze()
        self.model.cluster_model.unfreeze()
        self.model.fit(train_loader, val_loader, logger, self.args.train_cluster_net, centers=centers)
        self.model.cluster_model.freeze()
        self.clusters = self.model.get_clusters_centers()
        self._set_K(self.model.get_current_K())
        self.feature_extractor.unfreeze()
        self.feature_extractor.to(device=self.device)

    def freeze(self):
        self.model.cluster_model.freeze()
        self.feature_extractor.unfreeze()

    def unfreeze(self):
        self.model.cluster_model.unfreeze()
        self.model.cluster_model.to(device=self.device)

    def update_cluster_center_DPM(self, X, cluster_idx, assignments=None):
        n_samples = X.shape[0]
        for i in range(n_samples):
            if assignments[i, cluster_idx].item() > 0:
                self.count[cluster_idx] += assignments[i, cluster_idx].item()
                eta = 1.0 / self.count[cluster_idx]
                updated_cluster = (
                    (1 - eta) * self.clusters[cluster_idx]
                    + eta * X[i] * assignments[i, cluster_idx].item()
                )
                self.clusters[cluster_idx] = updated_cluster

    def update_cluster_center(self, X, cluster_idx, assignments=None):
        n_samples = X.shape[0]
        for i in range(n_samples):
            a = assignments[i, cluster_idx].item()
            if a > 0:
                self.count[cluster_idx] += a
                eta = 1.0 / self.count[cluster_idx]

                old_center = self.clusters[cluster_idx]
                new_point   = X[i]

                mus_list = [torch.Tensor(old_center), torch.Tensor(new_point)]
                N_i_list = [(1 - eta), eta * a]

                merged_mean = compute_merged_mean(mus_list, N_i_list)
                self.clusters[cluster_idx] = merged_mean.numpy()

    def update_cluster_covs(self, X, cluster_idx, assignments):
        return None

    def update_cluster_pis(self, X, cluster_idx, assignments):
        return None

    def update_assign(self, X, how_to_assign="min_dist"):
        """ Assign samples in `X` to clusters """
        if how_to_assign == "min_dist":
            return self._update_assign_min_dist(X.detach().cpu().numpy())
        elif how_to_assign == "forward_pass":
            return self.get_model_resp(X)

    def _update_assign_min_dist(self, X):
        dis_mat = self._compute_dist(X)
        hard_assign = np.argmin(dis_mat, axis=1)
        return self._to_one_hot(torch.tensor(hard_assign))

    def _to_one_hot(self, hard_assignments):
        return torch.nn.functional.one_hot(hard_assignments, num_classes=self.n_clusters)

    def _set_K(self, new_K):
        self.n_clusters = new_K
        self.count = 100 * np.ones((self.n_clusters))

    def get_model_params(self):
        mu, covs, pi, K = (
            self.model.get_clusters_centers(),
            self.model.get_clusters_covs(),
            self.model.get_clusters_pis(),
            self.n_clusters,
        )
        return mu, covs, pi, K

    def get_model_resp(self, codes):
        self.model.cluster_model.to(device=self.device)
        if self.args.regularization in ("cluster_loss", "cluster_loss_v2"):
            return self.model.cluster_model(codes, use_feature_extractor=False)
        else:
            with torch.no_grad():
                return self.model.cluster_model(codes, use_feature_extractor=False)
