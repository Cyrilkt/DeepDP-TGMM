#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import numpy as np
from joblib import Parallel, delayed
import torch
import torch.nn as nn
from src.clustering_models.clusternet_modules.clusternet_trainer import (
    ClusterNetTrainer,
)
from src.clustering_models.clusternet_modules.utils.clustering_utils.clustering_operations import compute_merged_mean


#def _parallel_compute_distance(X, cluster):
#    n_samples = X.shape[0]
#    dis_mat = np.zeros((n_samples, 1))
#    for i in range(n_samples):
#        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
#    return dis_mat

def cosinesimilarity(a,b,epsilon=1e-10):
    dot_product=np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (np.maximum(norm_a * norm_b, epsilon))
    
def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += 1-cosinesimilarity(X[i] ,cluster)
    return dis_mat


class ClusterNet(object):
    def __init__(self, args, feature_extractor):
        self.args = args
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters
        self.clusters = np.zeros((self.n_clusters, self.latent_dim))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = args.n_jobs
        self.feature_extractor = feature_extractor
        #self.device = "cuda" if torch.cuda.is_available() and args.gpus is not None else "cpu"
        self.device = f"cuda:{args.gpus}" if torch.cuda.is_available() and args.gpus is not None else "cpu"
    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i])
            for i in range(self.n_clusters)
        )
        dis_mat = np.hstack(dis_mat)

        return dis_mat

    def init_cluster(self, train_loader, val_loader, logger, indices=None, centers=None, init_num=0):
        """ Generate initial clusters using the clusternet
            init num is the number of time the clusternet was initialized (from the AE_ClusterPipeline module)
        """
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
        self.clusters = self.model.get_clusters_centers()  # copy clusters
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
        """ Update clusters centers on a batch of data

        Args:
            X (torch.tensor): All the data points that were assigned to this cluster
            cluster_idx (int): The cluster index
            assignments: The probability of each cluster to be assigned to this cluster (would be a vector of ones for hard assignment)
        """
        n_samples = X.shape[0]
        for i in range(n_samples):
            if assignments[i, cluster_idx].item() > 0:
                self.count[cluster_idx] += assignments[i, cluster_idx].item()
                eta = 1.0 / self.count[cluster_idx]
                updated_cluster = (1 - eta) * self.clusters[cluster_idx] + eta * X[i] * assignments[i, cluster_idx].item()
                # updated_cluster = (1 - eta) * self.clusters[cluster_idx] + eta * X[i]
                self.clusters[cluster_idx] = updated_cluster
    
    def update_cluster_center(self, X, cluster_idx, assignments=None):
        """ Update cluster centers on a batch of data, using spherical merging.
    
        Args:
            X (torch.tensor): All the data points that were assigned to this cluster
            cluster_idx (int): The cluster index
            assignments: Probability assignments of each point to this cluster (1D vector).
                         If None or all ones, this is essentially hard assignment.
        """
        n_samples = X.shape[0]
        for i in range(n_samples):
            a = assignments[i, cluster_idx].item()
            if a > 0:
                # Update the total count for this cluster
                self.count[cluster_idx] += a
                eta = 1.0 / self.count[cluster_idx]
    
                # Normalize the old cluster center and the new data point so they lie on the unit sphere
                old_center = self.clusters[cluster_idx]
                old_center = old_center #/ old_center.norm()
    
                new_point = X[i]
                new_point = new_point #/ new_point.norm()
    
                # We will merge just two 'clusters': 
                #  - The old cluster center with weight (1 - eta) 
                #  - The new point considered as a tiny 'cluster' with weight eta * a
                mus_list = [torch.Tensor(old_center), torch.Tensor(new_point)]
                N_i_list = [(1 - eta), eta * a]
    
                # Compute merged mean on the unit sphere
                merged_mean = compute_merged_mean(mus_list, N_i_list)
    
                # Update the cluster center
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
        """
        Takes LongTensor with index values of shape (*) and
        returns a tensor of shape (*, num_classes) that have zeros everywhere
        except where the index of last dimension matches the corresponding value
        of the input tensor, in which case it will be 1.
        """
        return torch.nn.functional.one_hot(hard_assignments, num_classes=self.n_clusters)

    def _set_K(self, new_K):
        self.n_clusters = new_K
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate, pseudo-counts

    def get_model_params(self):
        mu, covs, pi, K = self.model.get_clusters_centers(), self.model.get_clusters_covs(), self.model.get_clusters_pis(), self.n_clusters
        return mu, covs, pi, K

    def get_model_resp(self, codes):
        self.model.cluster_model.to(device=self.device)
        if self.args.regularization == "cluster_loss" or "cluster_loss_v2":
            # cluster assignment should have grad
            return self.model.cluster_model(codes,use_feature_extractor=False)
        else:
            # cluster assignment shouldn't have grad
            with torch.no_grad():
                return self.model.cluster_model(codes,use_feature_extractor=False)
