#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F 
from src.clustering_models.clusternet_modules.models.Classifiers import Subclustering_net
import math

from src.clustering_models.clusternet_modules.utils.clustering_utils.clustering_operations import (
    compute_pi_k,
    compute_mus,
    compute_covs,
    init_mus_and_covs_sub,
    compute_mus_covs_pis_subclusters,
    Log_mapping,
    cosine_dissimilarity_loss
)


class training_utils:
    def __init__(self, hparams):
        self.hparams = hparams
        self.pretraining_complete = False
        self.alt_count = 0
        self.last_performed = "merge"
        #self.device = "cuda" if torch.cuda.is_available() and hparams.gpus is not None else "cpu"
        self.device = f"cuda:{hparams.gpus}" if torch.cuda.is_available() and hparams.gpus is not None else "cpu"

    @staticmethod
    def change_model_requires_grad(model, require_grad_bool=True):
        for param in model.parameters():
            param.requires_grad = require_grad_bool

    @staticmethod
    def log_codes_and_responses(
        model_codes,
        model_gt,
        model_resp,
        model_resp_sub,
        codes,
        logits,
        y,
        sublogits=None,
        stage="train",
    ):
        """A function to log data used to compute model's parameters.

        Args:
            codes (torch.tensor): the current batch codes (in emedding space)
            logits (torch.tensor): the clustering net responses to the codes
            y (torch.tensor): the ground truth labels
            sublogits ([type], optional): [description]. Defaults to None. The subclustering nets response to the codes
        """
        if model_gt == []:
            # first batch of the epoch
            if codes is not None:
                model_codes = codes.detach().cpu()
            model_gt = y.detach().cpu()
            if logits is not None:
                model_resp = logits.detach().cpu()
            if sublogits is not None:
                model_resp_sub = sublogits.detach().cpu()
        else:
            if codes is not None:
                model_codes = torch.cat([model_codes, codes.detach().cpu()])
            model_gt = torch.cat([model_gt, y.detach().cpu()])
            if logits is not None:
                model_resp = torch.cat([model_resp, logits.detach().cpu()])
            if sublogits is not None:
                model_resp_sub = torch.cat([model_resp_sub, sublogits.detach().cpu()])
        return model_codes, model_gt, model_resp, model_resp_sub

    @staticmethod
    def log_vae_encodings(vae_means, vae_labels, means, labels):
        if vae_means == []:
            # start of an epoch
            vae_means = means.detach().cpu()
            vae_labels = labels.detach().cpu()
        else:
            vae_means = torch.cat([vae_means, means.detach().cpu()])
            vae_labels = torch.cat([vae_labels, labels.detach().cpu()])
        return vae_means, vae_labels

    def should_perform_split(self, current_epoch):
        # computes whether a split step should be performed in the current epoch
        #print("""Assessing if the current epoch condition with start_splitting etc are set to start a split validity checking """)
        #print('into should_perform_split')
        #print (self.last_performed)
        return (
            self.hparams.start_splitting <= current_epoch
            and (
                (current_epoch - self.hparams.start_splitting)
                % self.hparams.split_merge_every_n_epochs
                == 0
            )
            and self.last_performed == "merge"
        )

    def should_perform_merge(self, current_epoch, split_performed):
        # computes whether a merge step should be performed in the current epoch
        #print("""Assessing if the current epoch condition with start_merging etc are set to start a merged validity checking """)
        #print('into should_perform_merge')
        return (
            self.hparams.start_merging <= current_epoch
            and (
                (current_epoch - self.hparams.start_merging)
                % self.hparams.split_merge_every_n_epochs
                == 0
            )
            and not split_performed
            and self.last_performed == "split"
        )
    
    def freeze_mus_a_del(self, current_epoch, split_performed):
        """ VERSIONS OU ON REMOVE LA CONDITION SUR COMPUTE PARAMS EVERY"""
        print('current_epoch < self.hparams.start_computing_params :',current_epoch < self.hparams.start_computing_params)
        print('self.hparams.compute_params_every != 1',self.hparams.compute_params_every != 1)
        print('current_epoch % self.hparams.compute_params_every != 0',current_epoch % self.hparams.compute_params_every != 0)
        print('self.hparams.compute_params_every :',self.hparams.compute_params_every)
        if (
            current_epoch < self.hparams.start_computing_params
        ):
            print('INTO FREEZE MUS FIRST CDT')
            return True
        else:
            print('split occured freeze_mus', torch.tensor(
                    [
                        self.should_perform_split(current_epoch - epoch)
                        for epoch in range(
                            1,
                            self.hparams.freeze_mus_submus_after_splitmerge + 1,
                            1,
                        )
                    ]
                ))
            print('merge occured freeze_mus',torch.tensor(
                    [
                        self.should_perform_merge(
                            current_epoch - epoch, split_performed
                        )
                        for epoch in range(
                            1,
                            self.hparams.freeze_mus_submus_after_splitmerge + 1,
                            1,
                        )
                    ]
                ))
            
            split_occured = torch.tensor(
                    [
                        self.should_perform_split(current_epoch - epoch)
                        for epoch in range(
                            1,
                            self.hparams.freeze_mus_submus_after_splitmerge + 1,
                            1,
                        )
                    ]
                ).any()
            merge_occured = torch.tensor(
                    [
                        self.should_perform_merge(
                            current_epoch - epoch, split_performed
                        )
                        for epoch in range(
                            1,
                            self.hparams.freeze_mus_submus_after_splitmerge + 1,
                            1,
                        )
                    ]
                ).any()
            return split_occured or merge_occured
    def freeze_mus(self, current_epoch, split_performed):
        print('current_epoch < self.hparams.start_computing_params :',current_epoch < self.hparams.start_computing_params)
        print('self.hparams.compute_params_every != 1',self.hparams.compute_params_every != 1)
        print('current_epoch % self.hparams.compute_params_every != 0',current_epoch % self.hparams.compute_params_every != 0)
        print('self.hparams.compute_params_every :',self.hparams.compute_params_every)
        if (
            current_epoch < self.hparams.start_computing_params
            or (self.hparams.compute_params_every != 1 and current_epoch % self.hparams.compute_params_every != 0)
        ):
            print('INTO FREEZE MUS FIRST CDT')
            return True
        else:
            print('split occured freeze_mus', torch.tensor(
                    [
                        self.should_perform_split(current_epoch - epoch)
                        for epoch in range(
                            1,
                            self.hparams.freeze_mus_submus_after_splitmerge + 1,
                            1,
                        )
                    ]
                ))
            print('merge occured freeze_mus',torch.tensor(
                    [
                        self.should_perform_merge(
                            current_epoch - epoch, split_performed
                        )
                        for epoch in range(
                            1,
                            self.hparams.freeze_mus_submus_after_splitmerge + 1,
                            1,
                        )
                    ]
                ))
            
            split_occured = torch.tensor(
                    [
                        self.should_perform_split(current_epoch - epoch)
                        for epoch in range(
                            1,
                            self.hparams.freeze_mus_submus_after_splitmerge + 1,
                            1,
                        )
                    ]
                ).any()
            merge_occured = torch.tensor(
                    [
                        self.should_perform_merge(
                            current_epoch - epoch, split_performed
                        )
                        for epoch in range(
                            1,
                            self.hparams.freeze_mus_submus_after_splitmerge + 1,
                            1,
                        )
                    ]
                ).any()
            return split_occured or merge_occured

    def comp_cluster_params(self, train_resp, codes, pi, K,covs, prior=None):
        # compute pi
        print('INTO COMP_CLUSTER_PARAMS')
        
        pi = compute_pi_k(train_resp, prior=prior if self.hparams.use_priors else None)
        print('pi:',pi)
        
        mus = compute_mus(
            codes=codes,
            logits=train_resp,
            pi=pi,
            K=K,
            how_to_compute_mu=self.hparams.how_to_compute_mu,
            use_priors=self.hparams.use_priors,
            prior=prior,
            covs=covs,
        )
        print('into comp_cluster mus :',mus.size())

        new_covs = compute_covs(
            logits=train_resp,
            codes=codes,
            K=K,
            mus=mus,
            use_priors=self.hparams.use_priors,
            prior=prior)
        return pi, mus, new_covs
    
    
    

    def comp_subcluster_params(
        self,
        train_resp,
        train_resp_sub,
        codes,
        mus,
        K,
        n_sub_list,
        mus_sub,
        covs_sub,
        pi_sub,
        prior=None,
    ):
        print('COMP SUB ')
        mus_sub, covs_sub, pi_sub = compute_mus_covs_pis_subclusters(
            codes=codes, logits=train_resp, logits_sub=train_resp_sub,mus=mus,
            mus_sub=mus_sub,covs_sub=covs_sub, K=K, n_sub_list=n_sub_list, use_priors=self.hparams.use_priors, prior=prior
        )
        print('pi_sub',pi_sub)
        return pi_sub, mus_sub, covs_sub

    def init_subcluster_params_original(
        self, train_resp, train_resp_sub, codes, K, n_sub,mus ,prior=None
    ):
        mus_sub, covs_sub, pi_sub,n_sub_clusters = [], [], [],[]
        for k in range(K):
            mus_splt, covs, pis, n_sub_selected = init_mus_and_covs_sub(
                codes=codes,
                k=k,
                n_sub=n_sub,
                how_to_init_mu_sub=self.hparams.how_to_init_mu_sub,
                logits=train_resp,
                logits_sub=train_resp_sub,
                mus=mus,
                prior=prior,
                use_priors=self.hparams.use_priors,
                device=self.device
            )
            mus_splt=torch.stack([mus.to(self.device) for mus in mus_splt])
            n_sub_clusters.append(n_sub_selected)
            mus_sub.append(mus_splt)
            print('type covs',type(covs))
            covs_sub.append(covs)
            pi_sub.append(pis)
            print(' init subcluster params k:',k)
            #print('VOCS:',covs)
        print('N_SUB_CLUSTERS : ', n_sub_clusters)
        mus_sub = torch.cat(mus_sub)
        print('covs_sub :',covs_sub)
        #torch.save(covs_sub,'/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/covs_sub_kmeans.pt')
        if not self.hparams.ignore_subclusters:
            # initialize subclustering net
            codes_dim=codes.size(1)
            subclustering_net = Subclustering_net(self.hparams, codes_dim=codes_dim, k=K,subclusters_per_cluster=n_sub_clusters)
        else:
            subclustering_net = None
        if (self.hparams.how_to_init_mu_sub == 'kmeans_1d' or 'umap') and self.hparams.use_priors:
          #covs_sub=[torch.stack(covs_sub[i],dim=0) for i in range(len(covs_sub))]
          covs_sub = [torch.stack(covs_sub[i], dim=0) if isinstance(covs_sub[i], list) else covs_sub[i] for i in range(len(covs_sub))]

        covs_sub = torch.cat(covs_sub)
        pi_sub = torch.cat(pi_sub)

        return pi_sub, mus_sub, covs_sub,subclustering_net,n_sub_clusters
        
    def init_subcluster_params(
        self, train_resp, train_resp_sub, codes, K, n_sub, mus, prior=None
    ):
        mus_sub = []
        covs_sub = []
        pi_sub = []
        n_sub_clusters = []
    
        for k in range(K):
            # initialize subcluster mus, covs, pis, and count
            mus_splt, covs, pis, n_sub_selected = init_mus_and_covs_sub(
                codes=codes,
                k=k,
                n_sub=n_sub,
                how_to_init_mu_sub=self.hparams.how_to_init_mu_sub,
                logits=train_resp,
                logits_sub=train_resp_sub,
                mus=mus,
                prior=prior,
                use_priors=self.hparams.use_priors,
                device=self.device
            )
    
            # stack the new mus on device
            mus_splt = torch.stack([m.to(self.device) for m in mus_splt], dim=0)
            n_sub_clusters.append(n_sub_selected)
            mus_sub.append(mus_splt)
    
            # ——— Process covs: ensure each is a Tensor on the right device, then stack ———
            covs_tensors = []
            for c in covs:
                if not isinstance(c, torch.Tensor):
                    c = torch.as_tensor(c)
                covs_tensors.append(c.to(self.device))
            covs_tensor = torch.stack(covs_tensors, dim=0)  # shape (n_sub_selected, D, D)
            covs_sub.append(covs_tensor)
    
            # ——— Process pis: ensure it's a Tensor on the right device ———
            if isinstance(pis, list):
                pis_tensors = []
                for p in pis:
                    if not isinstance(p, torch.Tensor):
                        p = torch.as_tensor(p)
                    pis_tensors.append(p.to(self.device))
                pis = torch.stack(pis_tensors, dim=0)
            else:
                pis = pis.to(self.device)
            pi_sub.append(pis)
    
            print(f" init subcluster params k: {k}")
    
        print("N_SUB_CLUSTERS :", n_sub_clusters)
    
        # concatenate all mus and covs and pis
        mus_sub = torch.cat(mus_sub, dim=0)     # shape (K * n_sub_selected, D)
        covs_sub = torch.cat(covs_sub, dim=0)   # shape (K * n_sub_selected, D, D)
        pi_sub   = torch.cat(pi_sub,   dim=0)   # shape (K * n_sub_selected,)
    
        # initialize or skip subclustering net
        if not self.hparams.ignore_subclusters:
            codes_dim = codes.size(1)
            subclustering_net = Subclustering_net(
                self.hparams,
                codes_dim=codes_dim,
                k=K,
                subclusters_per_cluster=n_sub_clusters
            )
        else:
            subclustering_net = None
    
        return pi_sub, mus_sub, covs_sub, subclustering_net, n_sub_clusters

    
    def cluster_loss_function_hard_assign(
      self, c, r, model_mus, K, codes_dim, model_covs=None, pi=None, logger=None, warmup=False):
      """
      Computes a negative log-likelihood based cluster loss with hard assignments. 
      For each data point, only the cluster with the highest responsibility influences the loss.
  
      Parameters:
      - c (torch.Tensor): Data points (N, D).
      - r (torch.Tensor): Responsibility matrix (N, K).
      - model_mus (List[torch.Tensor]): List of mean vectors for each cluster.
      - K (int): Number of clusters.
      - codes_dim (int): Dimensionality of codes.
      - model_covs (List[torch.Tensor]): Covariance matrices for each cluster.
      - pi (torch.Tensor): Mixing coefficients (K,).
      - logger: Optional logger.
      - warmup (bool): If True, apply any warmup logic (not used here).
  
      Returns:
      - loss (torch.Tensor): Scalar loss value.
      """
      r=r.detach()
      device = self.device
      N = c.size(0)
      eps = 1e-10
  
      # Hard assignment: For each data point, pick the cluster with max responsibility
      assigned_cluster_indices = torch.argmax(r, dim=1)  # shape: (N,)
  
      # Create a one-hot encoding for the assigned clusters
      # shape: (N, K)
      assigned_one_hot = torch.zeros_like(r).scatter_(1, assigned_cluster_indices.unsqueeze(1), 1.0)
      
      log_probs = []
      for k in range(K):
          c_projected = Log_mapping(c, model_mus[k].to(device=device))
          gmm_k = MultivariateNormal(
              torch.zeros_like(model_mus[k], device=device), 
              model_covs[k].to(device=device)
          )
  
          log_prob_k = gmm_k.log_prob(c_projected)
          if pi is not None:
              log_prob_k = log_prob_k + torch.log(pi[k] + eps)
          
          log_probs.append(log_prob_k)
  
      # Stack log probabilities: shape: (N, K)
      log_probs = torch.stack(log_probs, dim=1)
  
      # Only the assigned cluster influences the loss
      # Hard assignment means sum_k assigned_one_hot[i,k] * log_probs[i,k] = log_probs[i,assigned_cluster]
      loss = -torch.mean(torch.sum(assigned_one_hot * log_probs, dim=1))
  
      return loss

    def cosine_dissimilarity_loss(x, centers, r, threshold=0.8, repulsion_weight=1):
      """
      Computes a contrastive cosine loss that attracts points to their assigned cluster center
      and repulses them from other cluster centers.
  
      Parameters:
      - x (torch.Tensor): Data points (N, D).
      - centers (torch.Tensor): Cluster centers (K, D).
      - r (torch.Tensor): Responsibilities (N, K).
      - threshold (float): Confidence threshold for filtering points.
      - repulsion_weight (float): Weight for the repulsion term.
  
      Returns:
      - loss (torch.Tensor): Combined loss value.
      """
      # Ensure all tensors are on the same device
      device = x.device
      centers = centers.to(device)
      r = r.detach().to(device)
  
      # Find the dominant cluster assignments
      assignments = torch.argmax(r, dim=1)  # (N,)
  
      # Filter points with r > threshold
      mask = r.max(dim=1).values > threshold  # (N,)
  
      # Select high-confidence data points
      x_high_conf = x[mask]  # (N_mask, D)
  
      # Apply the mask to assignments to filter valid centers
      filtered_assignments = assignments[mask]  # (N_mask,)
  
      # Select corresponding centers for high-confidence points
      selected_centers = centers[filtered_assignments]  # (N_mask, D)
  
      # Attraction Term: Cosine dissimilarity with assigned centers
      cos_sim_assigned = F.cosine_similarity(x_high_conf, selected_centers)
      attraction_loss = torch.mean(1 - cos_sim_assigned)
  
      # Repulsion Term: Cosine similarity with non-assigned centers
      repulsion_loss = 0.0
      if len(x_high_conf) > 0:  # Avoid empty masks
          for i in range(centers.size(0)):  # For each cluster center
              non_assigned_mask = filtered_assignments != i
              if non_assigned_mask.any():
                  other_centers = centers[i].unsqueeze(0).expand(len(x_high_conf), -1)  # Replicate center
                  cos_sim_other = F.cosine_similarity(x_high_conf[non_assigned_mask], other_centers)
                  repulsion_loss += torch.mean(cos_sim_other)
  
      # Normalize the repulsion loss
      repulsion_loss = repulsion_loss / (centers.size(0) - 1) if centers.size(0) > 1 else 0.0
  
      # Combine losses
      total_loss = attraction_loss + repulsion_weight * repulsion_loss
      return total_loss
    def target_assignment_loss(self,r):
      """
      Encourages the model to strengthen the assignment to the cluster with the highest initial responsibility.
      
      Parameters:
      - r (torch.Tensor): Responsibility matrix (N, K).
  
      Returns:
      - loss (torch.Tensor): Scalar loss value.
      """
      # Find the dominant cluster for each data point
      dominant_cluster = torch.argmax(r, dim=1)  # shape: (N,)
      
      # Create a one-hot encoded target matrix
      target = torch.zeros_like(r)
      target.scatter_(1, dominant_cluster.unsqueeze(1), 1.0)  # One-hot encoding
      
      # Encourage r to match the target distribution (e.g., using KL divergence)
      return F.kl_div(r.log(), target, reduction="batchmean")
    
    def cluster_loss_function_original(
        self, c, r, model_mus, K, codes_dim, model_covs=None, pi=None, logger=None,warmup=False):
        #if self.hparams.cluster_loss == "isotropic":
        if self.hparams.cluster_loss == "KL_GMM_2":
            r_gmm = []
            for k in range(K):
                c_projected=Log_mapping(c, model_mus[k].to(device=self.device))
                mu=torch.zeros(model_mus[k].size())
                gmm_k = MultivariateNormal(mu.to(device=self.device),model_covs[k].double().to(device=self.device))
                prob_k = gmm_k.log_prob(c_projected.detach().double())

                r_gmm.append((prob_k + torch.log(pi[k])).double())
            r_gmm = torch.stack(r_gmm).T
            max_values, _ = r_gmm.max(axis=1, keepdim=True)
            r_gmm -= torch.log(torch.exp((r_gmm - max_values)).sum(axis=1, keepdim=True)) + max_values
            r_gmm = torch.exp(r_gmm)
            #if warmup: # Adel
            #  threshold=0.2
            #  r_gmm = torch.where(r_gmm < threshold, r_gmm ** 2, r_gmm)
            eps = 0.00001
            r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1, keepdim=True)
            r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)
            
            
            return nn.KLDivLoss(reduction="batchmean")(
                torch.log(r),
                r_gmm.float().to(device=self.device),
            )
            
        elif self.hparams.cluster_loss == "KL_GMM_2_distord_log_mapping":
            # Step 1: Compute responsibilities (r_gmm)
            r_gmm = []
            for k in range(K):
                # Extract responsibilities for component k
                weights_k = r[:, k]  # Shape: (N,)
        
                # Apply weighted Log mapping
                c_projected = Log_mapping(c, model_mus[k].to(device=device), weights=weights_k, alpha=alpha)  # Shape: (N, D)
        
                # Define Gaussian component with mean zero in tangent space
                mu_zero = torch.zeros_like(model_mus[k]).to(device=device)  # Shape: (D,)
                gmm_k = MultivariateNormal(mu_zero.double(), model_covs[k].double().to(device=device))
        
                # Compute log probability
                prob_k = gmm_k.log_prob(c_projected.detach().double())  # Shape: (N,)
        
                # Append log probability plus log mixing coefficient
                r_gmm.append((prob_k + torch.log(pi[k])).double())  # List of (N,)
            
            # Stack responsibilities: Shape (N, K)
            r_gmm = torch.stack(r_gmm, dim=1)  # Shape: (N, K)
        
            # Log-sum-exp for numerical stability
            max_values, _ = r_gmm.max(dim=1, keepdim=True)  # Shape: (N, 1)
            r_gmm = r_gmm - max_values  # Shape: (N, K)
            r_gmm = r_gmm.exp()  # Shape: (N, K)
            r_gmm = r_gmm / r_gmm.sum(dim=1, keepdim=True)  # Normalize: Shape: (N, K)
            
        
            # Re-normalize responsibilities after thresholding
            r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(dim=1, keepdim=True)  # Shape: (N, K)
        
            # Ensure the target distribution r is also normalized
            r = (r + eps) / (r + eps).sum(dim=1, keepdim=True)  # Shape: (N, K)
        
            # Compute KL Divergence Loss
            kl_loss = nn.KLDivLoss(reduction="batchmean")(
                torch.log(r),
                r_gmm.float().to(device=device),
            )
        
            return kl_loss

        raise NotImplementedError("No such loss")
    
    
    
    

    def cluster_loss_function_cosine(
        self,
        c,              # [N, D] codes
        r,              # [N, K] soft responsibilities (logits or probs)
        model_mus,      # [K, D] centroids
        K,
        codes_dim,
        model_covs=None,
        pi=None,
        logger=None,
        warmup=False
    ):
        """
        Cosine-based cluster loss: global weighted mean of (1 - cos).
        Ensures all tensors are on the same device.
        """
        eps = 1e-6
        device = c.device
    
        # 1) Move everything to the right device
        r         = r.to(device)
        model_mus = model_mus.to(device)
    
        # 2) Normalize r into probabilities
        r_norm = r
        #r_norm = r_norm / r_norm.sum(dim=1, keepdim=True)  # [N, K]
    
        # 3) Normalize codes and centroids to unit length
        codes_norm = F.normalize(c,        p=2, dim=1, eps=eps)  # [N, D]
        mus_norm   = F.normalize(model_mus, p=2, dim=1, eps=eps)  # [K, D]
    
        # 4) Compute cosine distances
        #    Now both codes_norm and mus_norm are on `device`
        cosines = (codes_norm @ mus_norm.t()).clamp(-1 + eps, 1 - eps)  # [N, K]
        dists   = 1.0 - cosines                                        # [N, K]
    
        # 5) Weighted mean over all (i,k)
        loss = (r_norm * dists).mean()
    
        return loss
    


    
    def cluster_loss_function_cov(   # name kept for minimal changes
        self,
        c, r, model_mus, K, codes_dim,
        model_covs=None, pi=None,
        logger=None, warmup=False,
        parent_chunk_size=None, target_gpu_mem_frac: float = 0.75,
        covariance: str = "full",   # <— NEW
    ):
        """
        CHUNCK+covariance restriction
        Same API + covariance option ("full" | "diag" | "iso").
        Computes KL(r_gmm_eps || r_eps) with the same eps smoothing, streaming over K.
        """
        # helpers
        def _free_cuda_bytes():
            if torch.cuda.is_available():
                return int(torch.cuda.mem_get_info()[0])
            return 0
        def _auto_parent_chunk_size_fullcov(N, D, K, on_cuda: bool, dtype=torch.float64, target_gpu_mem_frac=0.25, safety=2.0):
            if K <= 0: return 1
            free_bytes = int(target_gpu_mem_frac * _free_cuda_bytes()) if on_cuda else 0
            if free_bytes <= 0: return max(1, min(K, 64))
            bytes_per = 8 if dtype == torch.float64 else 4
            per_component = N * (D + 1) * bytes_per  # rough upper bound for [N,Kc,D] + [N,Kc]
            Kc = int(free_bytes / (safety * max(per_component, 1)))
            return max(1, min(K, max(1, Kc)))
    
        device  = c.device
        on_cuda = (device.type == "cuda")
        dtype64 = torch.float64
        eps     = 1e-5
        LOG2PI  = math.log(2.0 * math.pi)
    
        # defaults
        if pi is None:
            pi = torch.full((K,), 1.0 / max(1, K), device=device, dtype=dtype64)
        if model_covs is None:
            D = model_mus.shape[-1]
            eye = torch.eye(D, device=device, dtype=dtype64)
            model_covs = eye.unsqueeze(0).repeat(K, 1, 1)
    
        # casts
        c64   = c.to(device=device, dtype=dtype64)
        mus64 = model_mus.to(device=device, dtype=dtype64)
        covs  = model_covs.to(device=device, dtype=dtype64)
        pi64  = pi.to(device=device, dtype=dtype64)
        r64   = r.to(device=device, dtype=dtype64)
    
        N, D = c64.shape
        assert mus64.shape == (K, D), f"model_mus must be (K,D); got {tuple(mus64.shape)}"
    
        # p normalization + logs
        pi64 = pi64 / pi64.sum()
        logpi = torch.log(pi64.clamp_min(1e-12))  # [K]
    
        # r_eps and its log
        r_eps     = (r64 + eps) / (r64 + eps).sum(dim=1, keepdim=True)  # [N,K]
        log_r_eps = torch.log(r_eps.clamp_min(1e-45))                   # [N,K]
    
        # chunk size
        if parent_chunk_size is None:
            parent_chunk_size = _auto_parent_chunk_size_fullcov(
                N=N, D=D, K=K, on_cuda=on_cuda, dtype=dtype64,
                target_gpu_mem_frac=target_gpu_mem_frac, safety=2.0
            )
    
        # Pass 1: streaming log-sum-exp to get logZ_i
        m = torch.full((N,), -float('inf'), dtype=dtype64, device=device)
        s = torch.zeros(N, dtype=dtype64, device=device)
    
        # Pre-extract compact representations if needed
        # (Avoid repeated diag/means inside the inner loop)
        if covariance == "full":
            covs_full = covs  # [K,D,D]
        elif covariance == "diag":
            if covs.ndim == 3 and covs.shape[-1] == D and covs.shape[-2] == D:
                covs_diag = torch.diagonal(covs, dim1=-2, dim2=-1)  # [K,D]
            else:
                assert covs.shape == (K, D), f"[diag] model_covs must be (K,D); got {tuple(covs.shape)}"
                covs_diag = covs
            covs_diag = covs_diag.clamp_min(1e-6)
        elif covariance == "iso":
            if covs.ndim == 3 and covs.shape[-1] == D and covs.shape[-2] == D:
                covs_iso = covs.diagonal(dim1=-2, dim2=-1).mean(dim=-1)      # [K]
            elif covs.ndim == 2 and covs.shape == (K, D):
                covs_iso = covs.mean(dim=-1)                                  # [K]
            else:
                assert covs.shape == (K,), f"[iso] model_covs must be (K,), got {tuple(covs.shape)}"
                covs_iso = covs
            covs_iso = covs_iso.clamp_min(1e-6)
        else:
            raise ValueError(f"Unknown covariance mode: {covariance}")
    
        for k0 in range(0, K, parent_chunk_size):
            k1 = min(k0 + parent_chunk_size, K)
            mus_chunk   = mus64[k0:k1]      # [Kc,D]
            logpi_chunk = logpi[k0:k1]      # [Kc]
    
            V_chunk = self.batched_Log_mapping(c64, mus_chunk)  # [N,Kc,D]
    
            if covariance == "full":
                covs_chunk = covs_full[k0:k1]                   # [Kc,D,D]
                zeros      = torch.zeros_like(mus_chunk)
                dist       = torch.distributions.MultivariateNormal(loc=zeros, covariance_matrix=covs_chunk)
                logp_chunk = dist.log_prob(V_chunk)             # [N,Kc]
            elif covariance == "diag":
                sigma2_c   = covs_diag[k0:k1]                   # [Kc,D]
                maha       = (V_chunk.pow(2) / sigma2_c.unsqueeze(0)).sum(dim=-1)          # [N,Kc]
                logdet     = torch.log(sigma2_c).sum(dim=-1).unsqueeze(0)                  # [1,Kc]
                const      = D * LOG2PI
                logp_chunk = -0.5 * (maha + logdet + const)                                 # [N,Kc]
            else:  # iso
                sigma2_c   = covs_iso[k0:k1]                    # [Kc]
                d2         = V_chunk.pow(2).sum(dim=-1)                                          # [N,Kc]
                logdet     = (D * torch.log(sigma2_c)).unsqueeze(0)                               # [1,Kc]
                const      = D * LOG2PI
                logp_chunk = -0.5 * (d2 / sigma2_c.unsqueeze(0) + logdet + const)                 # [N,Kc]
    
            logp_chunk = logp_chunk + logpi_chunk.unsqueeze(0)   # [N,Kc]
    
            # streaming LSE
            chunk_max = logp_chunk.max(dim=1).values
            m_new = torch.maximum(m, chunk_max)
            s = s * torch.exp(m - m_new) + torch.exp(logp_chunk - m_new.unsqueeze(1)).sum(dim=1)
            m = m_new
    
        logZ = m + torch.log(s.clamp_min(1e-300))  # [N]
    
        # Pass 2: KL(r_gmm_eps || r_eps)
        log_norm = math.log(1.0 + K * eps)
        sum_target_log_target = torch.zeros(N, dtype=dtype64, device=device)
        sum_target_log_input  = torch.zeros(N, dtype=dtype64, device=device)
    
        for k0 in range(0, K, parent_chunk_size):
            k1 = min(k0 + parent_chunk_size, K)
            mus_chunk   = mus64[k0:k1]
            logpi_chunk = logpi[k0:k1]
            log_r_eps_chunk = log_r_eps[:, k0:k1]               # [N,Kc]
    
            V_chunk = self.batched_Log_mapping(c64, mus_chunk)  # [N,Kc,D]
    
            if covariance == "full":
                covs_chunk = covs_full[k0:k1]
                zeros      = torch.zeros_like(mus_chunk)
                dist       = torch.distributions.MultivariateNormal(loc=zeros, covariance_matrix=covs_chunk)
                logp_chunk = dist.log_prob(V_chunk)
            elif covariance == "diag":
                sigma2_c   = covs_diag[k0:k1]
                maha       = (V_chunk.pow(2) / sigma2_c.unsqueeze(0)).sum(dim=-1)
                logdet     = torch.log(sigma2_c).sum(dim=-1).unsqueeze(0)
                const      = D * LOG2PI
                logp_chunk = -0.5 * (maha + logdet + const)
            else:
                sigma2_c   = covs_iso[k0:k1]
                d2         = V_chunk.pow(2).sum(dim=-1)
                logdet     = (D * torch.log(sigma2_c)).unsqueeze(0)
                const      = D * LOG2PI
                logp_chunk = -0.5 * (d2 / sigma2_c.unsqueeze(0) + logdet + const)
    
            logp_chunk = logp_chunk + logpi_chunk.unsqueeze(0)  # [N,Kc]
    
            soft_chunk = torch.exp(logp_chunk - logZ.unsqueeze(1))                  # [N,Kc]
            r_gmm_eps_chunk = (soft_chunk + eps) / (1.0 + K * eps)                  # [N,Kc]
    
            sum_target_log_target += (r_gmm_eps_chunk * (torch.log(soft_chunk + eps) - log_norm)).sum(dim=1)
            sum_target_log_input  += (r_gmm_eps_chunk * log_r_eps_chunk).sum(dim=1)
    
        KL_per_sample = sum_target_log_target - sum_target_log_input
        return KL_per_sample.mean().to(torch.float32)


    def cluster_loss_function_CHUNCK(
        self,
        c, r, model_mus, K, codes_dim,
        model_covs=None, pi=None,
        logger=None, warmup=False,
        parent_chunk_size=None, target_gpu_mem_frac: float = 0.25,
    ):
        """
        CHUNK VERSION
        Same API as your baseline.
        Computes KL(r_gmm_eps || r_eps) with identical eps-smoothing as the baseline,
        streaming over parents (K) to reduce memory.
        """
        # ---- helpers ----
        def _free_cuda_bytes():
            if torch.cuda.is_available():
                return int(torch.cuda.mem_get_info()[0])
            return 0
    
        def _auto_parent_chunk_size_fullcov(N, D, K, on_cuda: bool, dtype=torch.float64, target_gpu_mem_frac=0.25, safety=2.0):
            if K <= 0:
                return 1
            free_bytes = int(target_gpu_mem_frac * _free_cuda_bytes()) if on_cuda else 0
            if free_bytes <= 0:
                return max(1, min(K, 64))  # conservative default on CPU
            bytes_per = 8 if dtype == torch.float64 else 4
            # rough upper bound for [N,Kc,D] + [N,Kc]
            per_component = N * (D + 1) * bytes_per
            Kc = int(free_bytes / (safety * max(per_component, 1)))
            return max(1, min(K, max(1, Kc)))
    
        # ---- pick device from tensor c (always a torch.device) ----
        device  = c.device
        on_cuda = (device.type == "cuda")
        dtype64 = torch.float64
        eps     = 1e-5
    
        # ---- defaults ----
        if pi is None:
            pi = torch.full((K,), 1.0 / max(1, K), device=device, dtype=dtype64)
        if model_covs is None:
            D = model_mus.shape[-1]
            eye = torch.eye(D, device=device, dtype=dtype64)
            model_covs = eye.unsqueeze(0).repeat(K, 1, 1)
    
        # ---- cast / move ----
        c_64    = c.to(device=device, dtype=dtype64)
        mus_64  = model_mus.to(device=device, dtype=dtype64)
        covs_64 = model_covs.to(device=device, dtype=dtype64)
        pi_64   = pi.to(device=device, dtype=dtype64)
        r_64    = r.to(device=device, dtype=dtype64)
    
        N, D = c_64.shape
        assert mus_64.shape == (K, D), f"model_mus must be (K,D); got {tuple(mus_64.shape)}"
        assert covs_64.shape == (K, D, D), f"model_covs must be (K,D,D); got {tuple(covs_64.shape)}"
    
        # Normalize p; r will be eps-normalized below to match baseline exactly
        pi_64 = pi_64 / pi_64.sum()
        log_pi_64 = torch.log(pi_64.clamp_min(1e-12))
    
        # r_eps (baseline smoothing)
        r_eps = (r_64 + eps) / (r_64 + eps).sum(dim=1, keepdim=True)   # [N,K]
        log_r_eps = torch.log(r_eps.clamp_min(1e-45))                  # [N,K]
    
        # Choose chunk size
        if parent_chunk_size is None:
            parent_chunk_size = _auto_parent_chunk_size_fullcov(
                N=N, D=D, K=K, on_cuda=on_cuda, dtype=dtype64,
                target_gpu_mem_frac=target_gpu_mem_frac, safety=2.0
            )
    
        # ---- Pass 1: streaming log-sum-exp for logZ_i ----
        m = torch.full((N,), -float('inf'), dtype=dtype64, device=device)  # running max
        s = torch.zeros(N, dtype=dtype64, device=device)                   # running sum exp(logp - m)
    
        for k0 in range(0, K, parent_chunk_size):
            k1 = min(k0 + parent_chunk_size, K)
    
            mus_chunk   = mus_64[k0:k1]                  # [Kc,D]
            covs_chunk  = covs_64[k0:k1]                 # [Kc,D,D]
            logpi_chunk = log_pi_64[k0:k1]               # [Kc]
    
            # Log map -> [N,Kc,D]
            c_proj_chunk = self.batched_Log_mapping(c_64, mus_chunk)
    
            # log p_ik = log N(y | 0, S_k) + log p_k
            zeros_chunk = torch.zeros_like(mus_chunk)
            dist = torch.distributions.MultivariateNormal(loc=zeros_chunk, covariance_matrix=covs_chunk)
            logp_chunk = dist.log_prob(c_proj_chunk) + logpi_chunk.unsqueeze(0)  # [N,Kc]
    
            # streaming LSE
            chunk_max = logp_chunk.max(dim=1).values
            m_new = torch.maximum(m, chunk_max)
            s = s * torch.exp(m - m_new) + torch.exp(logp_chunk - m_new.unsqueeze(1)).sum(dim=1)
            m = m_new
    
        logZ = m + torch.log(s.clamp_min(1e-300))  # [N]
    
        # ---- Pass 2: KL(r_gmm_eps || r_eps) ----
        log_norm = math.log(1.0 + K * eps)  # same for all rows
        sum_target_log_target = torch.zeros(N, dtype=dtype64, device=device)
        sum_target_log_input  = torch.zeros(N, dtype=dtype64, device=device)
    
        for k0 in range(0, K, parent_chunk_size):
            k1 = min(k0 + parent_chunk_size, K)
    
            mus_chunk   = mus_64[k0:k1]
            covs_chunk  = covs_64[k0:k1]
            logpi_chunk = log_pi_64[k0:k1]
            log_r_eps_chunk = log_r_eps[:, k0:k1]  # [N,Kc]
    
            c_proj_chunk = self.batched_Log_mapping(c_64, mus_chunk)
            zeros_chunk = torch.zeros_like(mus_chunk)
            dist = torch.distributions.MultivariateNormal(loc=zeros_chunk, covariance_matrix=covs_chunk)
            logp_chunk = dist.log_prob(c_proj_chunk) + logpi_chunk.unsqueeze(0)  # [N,Kc]
    
            # softmax over k: soft_ik = exp(logp_ik - logZ_i)
            soft_chunk = torch.exp(logp_chunk - logZ.unsqueeze(1))              # [N,Kc]
    
            # r_gmm_eps chunk
            r_gmm_eps_chunk = (soft_chunk + eps) / (1.0 + K * eps)               # [N,Kc]
    
            # accumulate ? r_gmm_eps * log r_gmm_eps
            sum_target_log_target += (r_gmm_eps_chunk * (torch.log(soft_chunk + eps) - log_norm)).sum(dim=1)
            # accumulate ? r_gmm_eps * log r_eps
            sum_target_log_input  += (r_gmm_eps_chunk * log_r_eps_chunk).sum(dim=1)
    
        KL_per_sample = sum_target_log_target - sum_target_log_input
        return KL_per_sample.mean().to(torch.float32)



    
    def cluster_loss_function(
        self, c, r, model_mus, K, codes_dim, model_covs=None, pi=None,
        logger=None, warmup=False
    ):
        if self.hparams.cluster_loss != "KL_GMM_2":
            pass
    
        # 1) Riemannian log-map all codes onto each µ? at once: [N, K, D]
        #    (same as doing Log_mapping(c, µ?) for each k in a loop)
        c_proj = self.batched_Log_mapping(c.double(), model_mus.double())\
                         .to(self.device)  # [N, K, D]
    
        # 2) Build a *batched* zero-mean GMM over K components:
        zeros = torch.zeros_like(model_mus).double().to(self.device)    # [K, D]
        covs  = model_covs.double().to(self.device)                    # [K, D, D]
        gmm   = torch.distributions.MultivariateNormal(
                    loc=zeros,
                    covariance_matrix=covs
                )  # this is a batch of K Gaussians
    
        # 3) Compute log-prob for each sample under each component: [N, K]
        log_p = gmm.log_prob(c_proj)  # ?n,k: log ??(Log_map(c?), 0, S?)
    
        # 4) Add log-weights and do the usual log-sum-exp normalization
        log_p = log_p + torch.log(pi.to(self.device).double())        # [N, K]
        m, _ = log_p.max(dim=1, keepdim=True)                         # [N, 1]
        log_r_gmm = log_p - m - torch.logsumexp(log_p - m, dim=1, keepdim=True)
        r_gmm = torch.exp(log_r_gmm).float()                          # [N, K]
    
        # 5) Add eps and renormalize as you did
        eps = 1e-5
        r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(dim=1, keepdim=True)
        r     = (r     + eps) / (r     + eps).sum(dim=1, keepdim=True)
    
        # 6) Final batch-mean KL divergence
        return nn.KLDivLoss(reduction="batchmean")(
            torch.log(r),
            r_gmm.to(self.device),
        )

    def batched_Log_mapping(self,codes, mus):
        codes = codes.to(self.device)
        mus   = mus.to(self.device)

        # Normalize mus to ensure they lie on the unit sphere
        mus_normalized = F.normalize(mus, p=2, dim=-1)
        # Expand dimensions for broadcasting: [N, M, D]
        N, D = codes.shape
        M = mus_normalized.shape[0]
        codes_exp = codes.unsqueeze(1)  # [N, 1, D]
        mus_exp = mus_normalized.unsqueeze(0)  # [1, M, D]
        
        # Compute dot products and angles
        dot_products = torch.sum(codes_exp * mus_exp, dim=-1)  # [N, M]
        cos_theta = torch.clamp(dot_products, -1.0, 1.0)
        theta = torch.acos(cos_theta)
        
        # Avoid division by zero (replace sin(0)=0 with 1)
        sin_theta = torch.sin(theta)
        sin_theta = torch.where(sin_theta == 0, torch.ones_like(sin_theta), sin_theta)
        
        # Apply Riemannian logarithm mapping
        scaled_codes = (codes_exp - mus_exp * cos_theta.unsqueeze(-1))  # [N, M, D]
        proj_codes = scaled_codes / sin_theta.unsqueeze(-1) * theta.unsqueeze(-1)
        return proj_codes  # [N, M, D]
    
    
    
    def subcluster_loss_function_new_cosine(
        self,
        codes,                # [N, D]
        logits,               # [N, K]
        subresp,              # [N, M]  (M = sum(n_sub_list))
        K,
        n_sub_list,           # list of length K
        mus_sub,              # list of M tensors, each [D]
        covs_sub=None,
        pis_sub=None,
        cluster_labels=None,
        subcluster_labels=None,
        batch_idx=None,
        total_batches=None,
        masked_subresp=None):
        
        device = codes.device
        batch_size, D = codes.shape
        z = logits.argmax(-1)

        # Stack subcluster means [M, D]
        M = sum(n_sub_list)
        mus_sub_tensor = torch.stack([mu.to(device) for mu in mus_sub])  # [M, D]
        
        # 1) Flatten everything:
        C_tag   = codes.repeat(1, M).view(-1, D)        # [N*M, D]
        mus_tag = mus_sub_tensor.repeat(batch_size, 1)  # [N*M, D]
        r_tag   = subresp.flatten().clamp(min=1e-8)     # [N*M]
    
        # 2) Build parent-cluster index for each flattened subcluster:
        #    subcluster_parent[j] = parent cluster of subcluster j
        sub_parent = torch.cat([
            torch.full((n_sub,), k, device=device, dtype=torch.long)
            for k, n_sub in enumerate(n_sub_list)
        ])  # [M]
        parent_tag = sub_parent.repeat(batch_size)     # [N*M]
    
        # 3) Build code’s parent-cluster index for each flattened code:
        z_tag = z.repeat_interleave(M)                 # [N*M]
    
        # 4) Mask to keep only (n,j) where z_n == parent(j):
        mask = (z_tag == parent_tag)
    
        # 5) Compute cosine distances only on that mask:
        C_sel   = C_tag[mask]                          # [#valid, D]
        M_sel   = mus_tag[mask]                        # [#valid, D]
        r_sel   = r_tag[mask]                          # [#valid]
    
        cos_sim = F.cosine_similarity(C_sel, M_sel, dim=1)
        cos_dist= 1.0 - cos_sim
    
        # 6) Final weighted mean over valid pairs:
        return (r_sel * cos_dist).sum() / r_sel.sum()
        

    


    

    
    def subcluster_loss_function_new_previouspapier(
        self,
        codes,
        logits,
        subresp,
        K,
        n_sub_list,
        mus_sub,
        covs_sub=None,
        pis_sub=None,
        cluster_labels=None,
        subcluster_labels=None,
        batch_idx=None,
        total_batches=None,
        masked_subresp=None
    ):
        if self.hparams.subcluster_loss == "KL_GMM_2":
            device = codes.device
            batch_size, D = codes.shape
            total_num_subclusters = sum(n_sub_list)
            loss = 0.0
            z = logits.argmax(-1)  # [batch_size]
        
            # === Ensure all tensors are on correct device ===
            mus_sub = torch.stack([mu.to(device) for mu in mus_sub])  # [M, D]
            covs_sub_stack = torch.stack([
                cov.to(device) 
                for cov in covs_sub
            ])  # [M, D, D]
            pis_sub_tensor = torch.tensor(pis_sub, device=device, dtype=torch.float32).clamp(min=1e-12)
            log_pis = torch.log(pis_sub_tensor)  # [M]
        
            # === Vectorized Precomputation ===
            proj_codes_all = self.batched_Log_mapping(codes, mus_sub)  # [N, M, D]
            
            # Compute GMM log probabilities
            zero_means = torch.zeros(total_num_subclusters, D, device=device)  # [M, D]
            try:
                mvn = torch.distributions.MultivariateNormal(zero_means, covariance_matrix=covs_sub_stack)
                log_probs = mvn.log_prob(proj_codes_all)  # [N, M]
            except Exception as e:
                print(f"MVN failed: {e}")
                return torch.tensor(0.0, device=device)
            
            # Combine with mixture weights
            all_log_probs = log_probs + log_pis.unsqueeze(0)  # [N, M]
            all_log_probs[~torch.isfinite(all_log_probs)] = -float('inf')
            
            # === Parent Cluster Mapping ===
            subcluster_parent = torch.cat([
                torch.full((n_sub,), k, device=device, dtype=torch.long) 
                for k, n_sub in enumerate(n_sub_list)
            ])  # [M]
            
            # === Per-Cluster Processing ===
            for k in range(K):
                # Select data points in cluster k
                mask_k = (z == k)
                if not mask_k.any():
                    continue
                    
                # Select subclusters belonging to cluster k
                start_idx = sum(n_sub_list[:k])
                end_idx = start_idx + n_sub_list[k]
                mask_sub_k = slice(start_idx, end_idx)
                subcluster_indices = torch.arange(start_idx, end_idx, device=device)
                
                # Skip if no subclusters in this cluster
                if len(subcluster_indices) == 0:
                    continue
                    
                # Extract relevant log probabilities and responses
                logp_k = all_log_probs[mask_k, mask_sub_k]  # [N_k, M_k]
                r_k = subresp[mask_k, mask_sub_k]  # [N_k, M_k]
                
                # Filter valid subclusters (non -inf columns)
                col_valid = torch.any(torch.isfinite(logp_k), dim=0)  # [M_k]
                if not col_valid.any():
                    continue
                logp_k_valid = logp_k[:, col_valid]  # [N_k, M_valid]
                r_k_valid = r_k[:, col_valid].clamp(min=1e-8)  # [N_k, M_valid]
                
                # Filter valid data points (non -inf rows)
                row_valid = torch.any(torch.isfinite(logp_k_valid), dim=1)  # [N_k]
                if not row_valid.any():
                    continue
                logp_k_final = logp_k_valid[row_valid]  # [N_valid, M_valid]
                r_k_final = r_k_valid[row_valid]  # [N_valid, M_valid]
                
                # Compute KL divergence
                probs_gmm = torch.softmax(logp_k_final, dim=1)
                kl_loss_k = F.kl_div(torch.log(r_k_final), probs_gmm, reduction='batchmean')
                loss += kl_loss_k
            
            
        if self.hparams.subcluster_loss == "cosine":    
            """
            Fully vectorized cosine-distance subcluster loss. Uses same signature,
            computes expected cosine distance per parent cluster in one pass.
            """
            
            #subcluster_loss_function_new_cosinebatch
            device = codes.device
            N, D = codes.shape
            M = sum(n_sub_list)
            eps: float = 1e-6
            # Hard parent assignments
            z = logits.argmax(dim=-1)  # [N]
        
            # Stack centroids
            if isinstance(mus_sub, (list, tuple)):
                mus = torch.stack([mu.to(device) for mu in mus_sub], dim=0)  # [M, D]
            else:
                mus = mus_sub.to(device)
            
            # Normalize
            codes_norm = F.normalize(codes, p=2, dim=1, eps=eps)   # [N, D]
            mus_norm   = F.normalize(mus,   p=2, dim=1, eps=eps)   # [M, D]
        
            # Cosine similarity and distance
            cosines = codes_norm @ mus_norm.t()                   # [N, M]
            cosines = cosines.clamp(-1 + eps, 1 - eps)
            dists = 1 - cosines                                   # [N, M]
        
            # Build parent-subcluster mask matrix: shape [K, M]
            parent_idx = torch.arange(M, device=device)
            parent_map = torch.repeat_interleave(
                torch.arange(K, device=device), torch.tensor(n_sub_list, device=device)
            )  # [M]
            one_hot = F.one_hot(parent_map, num_classes=K).float()  # [M, K]
            mask = one_hot.t().unsqueeze(0)                        # [1, K, M]
            # Expand to [N, K, M]
            mask = mask.expand(N, -1, -1)
        
            # Expand dists and subresp to [N, K, M]
            dists_exp = dists.unsqueeze(1)       # [N, 1, M]
            r_exp     = subresp.unsqueeze(1)     # [N, 1, M]
        
            # Zero out irrelevant subclusters per parent
            dists_exp = dists_exp * mask         # [N, K, M]
            r_exp     = r_exp * mask             # [N, K, M]
        
            # Normalize r_exp along M dimension
            r_sum = r_exp.sum(dim=2, keepdim=True).clamp(min=eps)
            r_norm = r_exp / r_sum               # [N, K, M]
        
            # Expected distance per (i, k)
            exp_d = (r_norm * dists_exp).sum(dim=2)  # [N, K]
        
            # Mask points outside their parent cluster
            parent_mask = F.one_hot(z, num_classes=K).bool().unsqueeze(2)  # [N, K, 1]
            exp_d = exp_d * parent_mask.squeeze(2).float()                # [N, K]
        
            # Sum over k of mean_{i in cluster k}(exp_d[i,k])
            # To avoid loop, compute sum and counts
            counts = torch.bincount(z, minlength=K).clamp(min=1).float()   # [K]
            sum_d = exp_d.sum(dim=0)                                       # [K]
            loss = (sum_d / counts).sum()                                 # scalar
        
        """
        # === Parent Cluster Accuracy (Last Batch) ===
        if batch_idx is not None and total_batches is not None and batch_idx == total_batches - 1:
            with torch.no_grad():
                parent_log_lik = torch.full((batch_size, K), -float('inf'), device=device)
                for k in range(K):
                    mask_sub = (subcluster_parent == k)
                    if mask_sub.any():
                        log_probs_k = all_log_probs[:, mask_sub]  # [N, M_k]
                        parent_log_lik[:, k] = torch.logsumexp(log_probs_k, dim=1)
                
                inferred_parent = parent_log_lik.argmax(dim=1)
                acc = (inferred_parent == z).float().mean().item()
                print(f"Parent-cluster accuracy (last batch): {acc * 100:.2f}%")"""
          
    
        return 
    
    def subcluster_loss_function_new_covariance(self,
        codes, logits, subresp, K, n_sub_list, mus_sub, covs_sub, pis_sub,subcluster_labels=None,
        batch_idx=None,
        total_batches=None,
        masked_subresp=None,
        covariance: str = "full",   # NEW: "full" | "diag" | "iso"
    ):
        """
        Non-chunked original with covariance restriction inside the loss.
        Inputs: ambient full covs per subcluster (lists). We restrict to:
          - full: use full SPD cov
          - diag: keep only ambient diagonal
          - iso : use trace(S)/D
        Objective identical to your original:
            probs_gmm = softmax(logp_k)
            loss += KL( probs_gmm || r_k_clamped )
        """
        import math
        import torch
        import torch.nn.functional as F
        
        dtype = codes.dtype
        device = codes.device
        N, D = codes.shape
        M = sum(n_sub_list)
        LOG2PI = math.log(2.0 * math.pi)
    
        # Hard parent labels from logits
        z = logits.argmax(-1)  # [N]
    
        # === Ensure tensors on device, keep default dtype ===
        mus_sub = torch.stack([mu.to(device=device, dtype=dtype)  for mu in mus_sub])   # [M, D]
        covs_in = torch.stack([cv.to(device=device, dtype=dtype)  for cv in covs_sub])  # [M, D, D]
        pis_all = torch.as_tensor(pis_sub, device=device, dtype=dtype)                       # [M]
        log_pis = torch.log(pis_all.clamp_min(1e-12))                                        # [M]
    
        # === Symmetrize + tiny jitter (inside the loss only) ===
        covs_in = 0.5 * (covs_in + covs_in.transpose(-1, -2))
        covs_in = covs_in + (covs_in.new_ones((1, 1, 1)) * 1e-6).expand_as(covs_in)
    
        # === Vectorized Log-map (ambient ? tangent) ===
        proj_codes_all = self.batched_Log_mapping(codes, mus_sub)  # [N, M, D]
    
        # === Restrict covariance and compute log-likelihoods ===
        if covariance == "full":
            zeros = torch.zeros((M, D), device=device, dtype=dtype)
            mvn   = torch.distributions.MultivariateNormal(zeros, covariance_matrix=covs_in)
            logp_all = mvn.log_prob(proj_codes_all) + log_pis.unsqueeze(0)  # [N, M]
        elif covariance == "diag":
            sigma2  = torch.diagonal(covs_in, dim1=-2, dim2=-1).clamp_min(1e-6)  # [M, D]
            maha    = (proj_codes_all.pow(2) / sigma2.unsqueeze(0)).sum(dim=-1)  # [N, M]
            logdet  = torch.log(sigma2).sum(dim=-1).unsqueeze(0)                 # [1, M]
            const   = D * LOG2PI
            logp_all = -0.5 * (maha + logdet + const) + log_pis.unsqueeze(0)     # [N, M]
        elif covariance == "iso":
            sigma2  = torch.diagonal(covs_in, dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)  # [M]
            d2      = proj_codes_all.pow(2).sum(dim=-1)                         # [N, M]
            logdet  = (D * torch.log(sigma2)).unsqueeze(0)                      # [1, M]
            const   = D * LOG2PI
            logp_all = -0.5 * (d2 / sigma2.unsqueeze(0) + logdet + const) + log_pis.unsqueeze(0)
        else:
            raise ValueError(f"Unknown covariance mode: {covariance}")
    
        # Safety (should be finite in practice if S ? 0)
        logp_all = torch.where(torch.isfinite(logp_all), logp_all, logp_all.new_full((), -float('inf')))
    
        # === Per-parent accumulation (same as your original) ===
        total_loss = codes.new_tensor(0.0)
    
        start = 0
        for k in range(K):
            M_k = int(n_sub_list[k])
            end = start + M_k
            if M_k <= 0:
                start = end
                continue
    
            mask_k = (z == k)
            if not mask_k.any():
                start = end
                continue
    
            logp_k = logp_all[mask_k, start:end]     # [N_k, M_k]
            r_k    = subresp[mask_k,  start:end]     # [N_k, M_k]
    
            # Filter invalid columns/rows like your original
            col_valid = torch.any(torch.isfinite(logp_k), dim=0)
            if not col_valid.any():
                start = end
                continue
            logp_k = logp_k[:, col_valid]
            r_k    = r_k[:,  col_valid].clamp_min(1e-8)
    
            row_valid = torch.any(torch.isfinite(logp_k), dim=1)
            if not row_valid.any():
                start = end
                continue
            logp_k_final = logp_k[row_valid]
            r_k_final    = r_k[row_valid]
    
            # Same objective: KL( softmax(logp_k) || r_k )
            probs_gmm = torch.softmax(logp_k_final, dim=1)
            kl_k = F.kl_div(torch.log(r_k_final), probs_gmm, reduction='batchmean')
            total_loss = total_loss + kl_k
    
            start = end
    
        return total_loss.to(torch.float32)
    
    
    def subcluster_loss_function_new(
        self,
        codes,
        logits,
        subresp,
        K,
        n_sub_list,
        mus_sub,
        covs_sub=None,
        pis_sub=None,
        cluster_labels=None,
        subcluster_labels=None,
        batch_idx=None,
        total_batches=None,
        masked_subresp=None
    ):
        if self.hparams.subcluster_loss == "KL_GMM_2":
            device = codes.device
            batch_size, D = codes.shape
            total_num_subclusters = sum(n_sub_list)
            loss = 0.0
            z = logits.argmax(-1)  # [batch_size]
        
            # === Ensure all tensors are on correct device ===
            mus_sub = torch.stack([mu.to(device) for mu in mus_sub])  # [M, D]
            covs_sub_stack = torch.stack([
                cov.to(device) 
                for cov in covs_sub
            ])  # [M, D, D]
            pis_sub_tensor = torch.tensor(pis_sub, device=device, dtype=torch.float32).clamp(min=1e-12)
            log_pis = torch.log(pis_sub_tensor)  # [M]
        
            # === Vectorized Precomputation ===
            proj_codes_all = self.batched_Log_mapping(codes, mus_sub)  # [N, M, D]
            
            # Compute GMM log probabilities
            zero_means = torch.zeros(total_num_subclusters, D, device=device)  # [M, D]
            try:
                mvn = torch.distributions.MultivariateNormal(zero_means, covariance_matrix=covs_sub_stack)
                log_probs = mvn.log_prob(proj_codes_all)  # [N, M]
            except Exception as e:
                print(f"MVN failed: {e}")
                return torch.tensor(0.0, device=device)
            
            # Combine with mixture weights
            all_log_probs = log_probs + log_pis.unsqueeze(0)  # [N, M]
            all_log_probs[~torch.isfinite(all_log_probs)] = -float('inf')
            
            # === Parent Cluster Mapping ===
            subcluster_parent = torch.cat([
                torch.full((n_sub,), k, device=device, dtype=torch.long) 
                for k, n_sub in enumerate(n_sub_list)
            ])  # [M]
            
            # === Per-Cluster Processing ===
            for k in range(K):
                # Select data points in cluster k
                mask_k = (z == k)
                if not mask_k.any():
                    continue
                    
                # Select subclusters belonging to cluster k
                start_idx = sum(n_sub_list[:k])
                end_idx = start_idx + n_sub_list[k]
                mask_sub_k = slice(start_idx, end_idx)
                subcluster_indices = torch.arange(start_idx, end_idx, device=device)
                
                # Skip if no subclusters in this cluster
                if len(subcluster_indices) == 0:
                    continue
                    
                # Extract relevant log probabilities and responses
                logp_k = all_log_probs[mask_k, mask_sub_k]  # [N_k, M_k]
                r_k = subresp[mask_k, mask_sub_k]  # [N_k, M_k]
                
                # Filter valid subclusters (non -inf columns)
                col_valid = torch.any(torch.isfinite(logp_k), dim=0)  # [M_k]
                if not col_valid.any():
                    continue
                logp_k_valid = logp_k[:, col_valid]  # [N_k, M_valid]
                r_k_valid = r_k[:, col_valid].clamp(min=1e-8)  # [N_k, M_valid]
                
                # Filter valid data points (non -inf rows)
                row_valid = torch.any(torch.isfinite(logp_k_valid), dim=1)  # [N_k]
                if not row_valid.any():
                    continue
                logp_k_final = logp_k_valid[row_valid]  # [N_valid, M_valid]
                r_k_final = r_k_valid[row_valid]  # [N_valid, M_valid]
                
                # Compute KL divergence
                probs_gmm = torch.softmax(logp_k_final, dim=1)
                kl_loss_k = F.kl_div(torch.log(r_k_final), probs_gmm, reduction='batchmean')
                loss += kl_loss_k
            
            
        elif self.hparams.subcluster_loss == "cosine":
            device = codes.device
            N, D = codes.shape
            K_cur = logits.shape[1]                 # <-- do NOT trust passed-in K
            eps = 1e-8
        
            # hard parent assignments
            z = logits.argmax(dim=-1).long()        # [N]
            # quick guards
            assert (z >= 0).all() and (z < K_cur).all(), "z has out-of-range labels"
        
            # stack sub-centroids
            mus = (torch.stack(mus_sub, dim=0) if isinstance(mus_sub, (list, tuple)) else mus_sub).to(device)
            M = mus.shape[0]
            assert subresp.shape == (N, M), f"subresp must be [N,M]; got {subresp.shape}"
        
            # L2-normalize
            codes_n = F.normalize(codes, p=2, dim=1)
            mus_n   = F.normalize(mus,   p=2, dim=1)
        
            # spherical isotropic distance: ||x-µ||^2 = 2(1 - cos)
            cos   = (codes_n @ mus_n.t()).clamp(-1 + eps, 1 - eps)   # [N,M]
            dists = 2.0 * (1.0 - cos)                                # [N,M]
        
            # parent map for each subcluster (length M, values in [0..K_cur-1])
            assert len(n_sub_list) == K_cur, "n_sub_list length must match current K (after split/merge)"
            parent_map = torch.repeat_interleave(
                torch.arange(K_cur, device=device),
                torch.tensor(n_sub_list, device=device)
            )                                                        # [M]
            assert parent_map.max() < K_cur and parent_map.min() >= 0
        
            # mask [K_cur, M] to keep only subclusters of each parent
            mask_KM = F.one_hot(parent_map, num_classes=K_cur).T.float()  # [K_cur, M]
        
            # Expected distance E_{s|k} per (i,k)
            w   = subresp.clamp_min(eps)            # [N,M]
            num = (w * dists) @ mask_KM.T           # [N,K_cur]  sum over subclusters of k
            den = w @ mask_KM.T                     # [N,K_cur]
            exp_d = num / den.clamp_min(eps)        # [N,K_cur]
        
            # pick only the parent actually assigned to each i, then average over i
            loss = exp_d.gather(1, z.view(-1,1)).mean()

        elif self.hparams.subcluster_loss == "geo_logmap_sq":
            device = codes.device
            N, D = codes.shape
            K_cur = logits.shape[1]               # dynamic K after split/merge
            eps = 1e-8
        
            # hard parent assignments
            z = logits.argmax(dim=-1).long()      # [N]
            assert (z >= 0).all() and (z < K_cur).all()
        
            # centroids (M total subclusters)
            mus = (torch.stack(mus_sub, dim=0) if isinstance(mus_sub, (list, tuple)) else mus_sub).to(device)
            M = mus.shape[0]
            assert subresp.shape == (N, M), f"subresp must be [N,M], got {subresp.shape}"
            assert len(n_sub_list) == K_cur, "n_sub_list length must match current K"
        
            # L2-normalize to lie on the sphere
            x_n = F.normalize(codes, p=2, dim=1)      # [N,D]
            mu_n = F.normalize(mus,   p=2, dim=1)     # [M,D]
        
            # cos and geodesic angle
            cos = (x_n @ mu_n.t()).clamp(-1 + eps, 1 - eps)  # [N,M]
            theta = torch.acos(cos)                           # [N,M]
            theta2 = theta * theta                            # [N,M]  == ||Log_mu(x)||^2
        
            # parent map [M] with values in [0..K_cur-1], mask [K_cur,M]
            parent_map = torch.repeat_interleave(
                torch.arange(K_cur, device=device),
                torch.tensor(n_sub_list, device=device)
            )                                                 # [M]
            mask_KM = F.one_hot(parent_map, num_classes=K_cur).T.float()  # [K_cur, M]
        
            # expected geodesic^2 within parent: E_{s|k}[theta^2]
            w   = subresp.clamp_min(eps)                      # [N,M]
            num = (w * theta2) @ mask_KM.T                    # [N,K_cur]
            den = w @ mask_KM.T                               # [N,K_cur]
            exp_d = num / den.clamp_min(eps)                  # [N,K_cur]
        
            # pick assigned parent per sample and average over samples (isotropic MSE on sphere)
            loss = exp_d.gather(1, z.view(-1,1)).mean()

        """
        # === Parent Cluster Accuracy (Last Batch) ===
        if batch_idx is not None and total_batches is not None and batch_idx == total_batches - 1:
            with torch.no_grad():
                parent_log_lik = torch.full((batch_size, K), -float('inf'), device=device)
                for k in range(K):
                    mask_sub = (subcluster_parent == k)
                    if mask_sub.any():
                        log_probs_k = all_log_probs[:, mask_sub]  # [N, M_k]
                        parent_log_lik[:, k] = torch.logsumexp(log_probs_k, dim=1)
                
                inferred_parent = parent_log_lik.argmax(dim=1)
                acc = (inferred_parent == z).float().mean().item()
                print(f"Parent-cluster accuracy (last batch): {acc * 100:.2f}%")"""
          
    
        return loss
    
    

    def subcluster_loss_function_new_v22(  
        self, 
        codes, 
        logits, 
        subresp, 
        K, 
        n_sub_list, 
        mus_sub, 
        covs_sub=None, 
        pis_sub=None, 
        cluster_labels=None, 
        subcluster_labels=None, 
        batch_idx=None, 
        total_batches=None,
        masked_subresp=None
    ):
        if self.hparams.subcluster_loss == "cosine_dissimilarity":
            # (not implemented here)
            pass
    
        elif self.hparams.subcluster_loss == "KL_GMM_2":
            loss = 0.0
            subcluster_offset = 0
            z = logits.argmax(-1)  # [batch_size]
    
            # Only debug-print the last batch
            if (
                masked_subresp is not None 
                and batch_idx is not None 
                and total_batches is not None 
                and batch_idx == total_batches - 1
            ):
                print("\n=== Debugging Subcluster Masking (last batch) ===")
                for idx in range(codes.size(0)):
                    main_cluster = z[idx].item()
                    offset = sum(n_sub_list[:main_cluster])
                    num_subs = n_sub_list[main_cluster]
                    try:
                        selected_subs = masked_subresp[idx, offset : offset + num_subs]
                    except IndexError:
                        print(f"Data Point {idx}: IndexError check offsets.")
                        continue
                    non_selected = masked_subresp[idx].clone()
                    non_selected[offset : offset + num_subs] = -float("inf")
                    ok_sel = torch.all(selected_subs != -float("inf"))
                    ok_non = torch.all(non_selected == -float("inf"))
                    if not (ok_sel and ok_non):
                        print(f"Point {idx}: Masking error.")
                        if not ok_non:
                            print("  Non-selected not all -inf.")
                        if not ok_sel:
                            print("  Selected contains -inf.")
                        print(f"  Mask:\n{masked_subresp[idx]}\n")
                    else:
                        print(f"Point {idx}: Masking OK.")
                print("=== End Mask Debug ===\n")
    
            total_num_subclusters = sum(n_sub_list)
            all_log_probs = torch.full(
                (codes.size(0), total_num_subclusters),
                -float("inf"),
                device=self.device,
                dtype=torch.float32
            )
            subcluster_parent = []
    
            for k in range(K):
                n_sub_k = n_sub_list[k]
                mask_k = (z == k)
                codes_k = codes[mask_k]  # only points in cluster k
                r = subresp[mask_k, subcluster_offset : subcluster_offset + n_sub_k]
                valid_subs = []
    
                if codes_k.numel() > 0:
                    r_gmm_list = []
    
                    if batch_idx == total_batches - 1:
                        lab = cluster_labels[k] if cluster_labels else f"Cluster {k}"
                        print(f"Processing cluster '{lab}' (index {k})")
    
                    for k_sub in range(n_sub_k):
                        sub_idx = subcluster_offset + k_sub
    
                        # -- MINIMAL CHANGE: force mu, cov, pi_val onto GPU as float32 --
                        mu = mus_sub[sub_idx].float().to(self.device)
                        cov = covs_sub[sub_idx].float().to(self.device)
                        pi_val = pis_sub[sub_idx].float().to(self.device)
                        log_pi = torch.log(pi_val)
                        if torch.isinf(log_pi).item():
                            continue
                        # ------------------------------------------------------------
    
                        # Build projected codes for just the points in cluster k
                        proj_codes_k = Log_mapping(codes_k, mu)
                        proj_mu = torch.zeros_like(mu)  # float32 on GPU
    
                        gmm_k = torch.distributions.MultivariateNormal(
                            proj_mu, 
                            covariance_matrix=cov
                        )
                        logp_k = gmm_k.log_prob(proj_codes_k)  # [num_in_k]
                        r_gmm_list.append(logp_k + log_pi)
                        valid_subs.append(k_sub)
    
                        # Fill “all_log_probs” for EVERY point in the batch
                        proj_codes_all = Log_mapping(codes, mu)  # mu on GPU
                        gmm_all = torch.distributions.MultivariateNormal(
                            proj_mu,
                            covariance_matrix=cov
                        )
                        logp_all = gmm_all.log_prob(proj_codes_all) + log_pi
                        if torch.isnan(logp_all).any():
                            raise ValueError(f"NaN in prob_all at subcluster {sub_idx}")
                        all_log_probs[:, sub_idx] = logp_all
                        subcluster_parent.append(k)
    
                    if len(r_gmm_list) == 0:
                        subcluster_offset += n_sub_k
                        print(f"  Skipping cluster {k}: all p were too small.")
                        continue
    
                    # Stack per-subcluster log-probs for points in k
                    r_gmm_mat = torch.stack(r_gmm_list, dim=1)  # [#in_k, #valid_subs]
                    max_vals, _ = r_gmm_mat.max(dim=1, keepdim=True)
                    normed = r_gmm_mat - max_vals
                    log_sum = (normed.exp().sum(dim=1, keepdim=True)).log()
                    r_gmm_norm = (normed - log_sum).exp()
                    eps = 1e-5
                    r_gmm_norm = (r_gmm_norm + eps) / (r_gmm_norm + eps).sum(dim=1, keepdim=True)
    
                    if torch.isnan(r_gmm_norm).any():
                        raise ValueError(f"NaN in r_gmm_norm for cluster {k}")
    
                    r_sub_sel = r[:, valid_subs]
                    kl_loss_k = nn.KLDivLoss(reduction="batchmean")(
                        torch.log(r_sub_sel),
                        r_gmm_norm.float().to(device=self.device)
                    )
                    loss += kl_loss_k
    
                subcluster_offset += n_sub_k
    
            # Final-batch check of parent-cluster inference
            if batch_idx == total_batches - 1:
                with torch.no_grad():
                    parent_tensor = torch.tensor(subcluster_parent, device=self.device, dtype=torch.long)
                    parent_log_lik = torch.full(
                        (codes.size(0), K),
                        -float("inf"),
                        device=self.device,
                        dtype=torch.float32
                    )
                    for k in range(K):
                        idxs = (parent_tensor == k).nonzero(as_tuple=True)[0]
                        if idxs.numel() == 0:
                            continue
                        parent_log_lik[:, k] = torch.logsumexp(all_log_probs[:, idxs], dim=1)
    
                    inferred_parent = parent_log_lik.argmax(dim=1)
                    corr = (inferred_parent == z).float().mean().item()
                    print(f"Parent-cluster accuracy (last batch): {corr * 100:.2f}%")
    
            return loss
    
        else:
            raise NotImplementedError(f"Unknown subcluster_loss: {self.hparams.subcluster_loss}")






    def subcluster_loss_function_new_todecoch( 
        self, 
        codes, 
        logits, 
        subresp, 
        K, 
        n_sub_list, 
        mus_sub, 
        covs_sub=None, 
        pis_sub=None, 
        cluster_labels=None, 
        subcluster_labels=None, 
        batch_idx=None, 
        total_batches=None,
        masked_subresp=None  # New optional parameter
    ):
        """
        Calculates the subcluster loss with optional debugging to verify subcluster assignments.
    
        Args:
            codes (Tensor): Input codes.
            logits (Tensor): Logits for main clusters.
            subresp (Tensor): Subcluster responses (probabilities).
            K (int): Number of main clusters.
            n_sub_list (List[int]): Number of subclusters per main cluster.
            mus_sub (List[Tensor]): Means for subclusters.
            covs_sub (List[Tensor], optional): Covariances for subclusters.
            pis_sub (List[Tensor], optional): Mixing coefficients for subclusters.
            cluster_labels (List[str], optional): Labels for main clusters.
            subcluster_labels (List[List[str]], optional): Labels for subclusters.
            batch_idx (int, optional): Current batch index.
            total_batches (int, optional): Total number of batches.
            masked_subresp (Tensor, optional): Masked subcluster responses for debugging.
    
        Returns:
            Tensor: Calculated loss.
        """
        # Check if subcluster loss is to be calculated
        if self.hparams.subcluster_loss == "cosine_dissimilarity":
            # Implement cosine dissimilarity loss if applicable
            pass
        elif self.hparams.subcluster_loss == "KL_GMM_2":
            loss = 0.0
            subcluster_offset = 0  # To track subcluster index offsets
    
            z = logits.argmax(-1)  # Original parent cluster assignments
    
            # === Debugging Section ===
            # Verify subcluster masking if masked_subresp is provided
            if masked_subresp is not None and batch_idx is not None and total_batches is not None:
                if batch_idx == total_batches - 1:  # Perform checks only on the last batch
                    print("\n=== Debugging Subcluster Masking ===")
                    for idx in range(codes.size(0)):  # Iterate over each data point in the batch
                        # Get the main cluster assignment for the current data point
                        main_cluster = z[idx].item()
    
                        # Determine the range of subclusters for this main cluster
                        # Use built-in sum() to ensure native Python integers
                        cluster_offset = sum(n_sub_list[:main_cluster])
                        num_subclusters = n_sub_list[main_cluster]
    
                        # Extract selected and non-selected subclusters
                        try:
                            selected_subclusters = masked_subresp[idx, cluster_offset : cluster_offset + num_subclusters]
                        except IndexError:
                            print(f"Data Point {idx}: IndexError - Check cluster_offset and num_subclusters.")
                            continue  # Skip to the next data point
    
                        # Clone the entire row and set selected subclusters to -inf to get non-selected subclusters
                        non_selected_subclusters = masked_subresp[idx].clone()
                        non_selected_subclusters[cluster_offset : cluster_offset + num_subclusters] = -float('inf')
    
                        # Check conditions
                        non_selected_correct = torch.all(non_selected_subclusters == -float('inf'))
                        selected_correct = torch.all(selected_subclusters != -float('inf'))
    
                        # Print debug information based on the checks
                        if not non_selected_correct or not selected_correct:
                            print(f"Data Point {idx}: Incorrect subcluster masking detected.")
                            if not non_selected_correct:
                                print(f"  Non-selected subclusters are not all -inf.")
                            if not selected_correct:
                                print(f"  Selected subclusters contain -inf values.")
                            print(f"  Mask:\n{masked_subresp[idx]}\n")
                        else:
                            print(f"Data Point {idx}: Subcluster masking is correct.")
                    print("=== End of Debugging Subcluster Masking ===\n")
    
            # Calculate the total number of subclusters across all parent clusters
            total_num_subclusters = sum(n_sub_list)
    
            # Initialize a tensor to store log-likelihoods for each data point across all subclusters
            # Shape: (batch_size, total_num_subclusters)
            all_log_probs = torch.full(
                (codes.size(0), total_num_subclusters), 
                -float('inf'), 
                device=self.device
            )
    
            # Initialize a list to map each subcluster to its parent cluster
            # Length: total_num_subclusters
            subcluster_parent = []
    
            # Iterate over each main cluster
            
            for k in range(K):
                # Get the number of subclusters for cluster k
                n_sub_k = n_sub_list[k]
                codes_k = codes[z == k]
                r = subresp[z == k, subcluster_offset : subcluster_offset + n_sub_k]
                # Initialize a list to keep track of valid subcluster indices
                valid_subcluster_indices = []
                
                if len(codes_k) > 0:
                    r_gmm = []
                    cluster_label = cluster_labels[k] if cluster_labels else f"Cluster {k}"
                    if batch_idx == total_batches - 1:
                        print(f"Processing cluster '{cluster_label}' (MLP index {k})")
                    for k_sub in range(n_sub_k):
                        subcluster_idx = subcluster_offset + k_sub
                        mu = mus_sub[subcluster_idx].double().to(device=self.device)
    
                        # Existing computations...
                        proj_codes = Log_mapping(codes_k.detach().double(), mu)
                        proj_mu = torch.zeros(mu.size()).to(device=self.device)
                        
                        log_pis_sub = torch.log(pis_sub[subcluster_idx]) 
                        if torch.isinf(log_pis_sub).item():  # Threshold in log-space, equivalent to pis_sub <= exp(-10)
                            print(f"Skipping subcluster {k_sub} (subcluster_idx={subcluster_idx}) due to small log(pis_sub): {log_pis_sub}")
                            continue
                        # Check for NaN in proj_codes and proj_mu
                        if torch.isnan(proj_codes).any():
                            raise ValueError(f"NaN detected in proj_codes for subcluster {k_sub}. proj_codes: {proj_codes}")
                        if torch.isnan(proj_mu).any():
                            raise ValueError(f"NaN detected in proj_mu for subcluster {k_sub}. proj_mu: {proj_mu}")
                        valid_subcluster_indices.append(k_sub)
                        # Correctly indexing covs_sub
                        gmm_k = MultivariateNormal(
                            proj_mu, 
                            covs_sub[subcluster_idx].double().to(device=self.device)
                        )
                        prob_k = gmm_k.log_prob(proj_codes)
                        #if batch_idx == total_batches - 1:
                        #    print(f"Range of proj_codes: min={proj_codes.min().item()}, max={proj_codes.max().item()}")
                        if torch.isnan(prob_k).any():
                            raise ValueError(f"NaN detected in prob_k for subcluster {k_sub}. prob_k: {prob_k}")
                        r_gmm.append((prob_k + torch.log(pis_sub[subcluster_idx])).double())
                        #### debug
                        #print(' RGMM COMPONENT COMPUTATION')
                        #print(' subcluster_idx :', subcluster_idx)
                        cov_eigenvalues = torch.linalg.eigvalsh(covs_sub[subcluster_idx].double())
                        #print('(cov_eigenvalues <= 0).any() :',(cov_eigenvalues <= 0).any())
                        #print('prob_k :',prob_k)
                        #print('torch.log(pis_sub[subcluster_idx]) :',torch.log(pis_sub[subcluster_idx]))
                        #print('r_gmm :',r_gmm)
    
                        # Compute log-probability for all data points (not just codes_k)
                        proj_codes_all = Log_mapping(codes.detach().double(), mu)
                        
                        gmm_all = MultivariateNormal(
                            proj_mu, 
                            covs_sub[subcluster_idx].double().to(device=self.device)
                        )
                        prob_all = gmm_all.log_prob(proj_codes_all) + torch.log(pis_sub[subcluster_idx])
                        # Check for NaN in prob_all
                        if torch.isnan(prob_all).any():
                            raise ValueError(f"NaN detected in prob_all for subcluster {k_sub}. prob_all: {prob_all}")
                            
                        all_log_probs[:, subcluster_idx] = prob_all
    
                        # Map subcluster to its parent cluster
                        subcluster_parent.append(k)
    
                    # Stack and normalize r_gmm
                    if len(r_gmm) == 0:
                       # Update the subcluster offset
                       subcluster_offset += n_sub_k
                       print(f" Passing Cluster {k} subcluster loss computation due to empty subclusters")
                       continue
                    r = r[:, valid_subcluster_indices]
                    r_gmm = torch.stack(r_gmm).T  # Shape: (batch_size_k, n_sub_k)
                    max_values, _ = r_gmm.max(dim=1, keepdim=True)
                    r_gmm = r_gmm - torch.log(torch.exp(r_gmm - max_values).sum(dim=1, keepdim=True)) - max_values
                    r_gmm = torch.exp(r_gmm)
                    # Check for NaN in r_gmm
                    # Check for NaN in r_gmm
                    if torch.isnan(r_gmm).any():
                        print('proj_codes :',proj_codes)
                        print('proj_mu :',proj_mu)
                        print('prob_k :',prob_k)
                        print('prob_all :',prob_all)
                        raise ValueError(f"NaN detected in r_gmm for cluster {k}. r_gmm: {r_gmm}")
                    eps = 1e-5
                    r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(dim=1, keepdim=True)
    
                    #if batch_idx == total_batches - 1:
                    #    kl_div = nn.KLDivLoss(reduction="batchmean")(
                    #        torch.log(r),
                    #        r_gmm.float().to(device=self.device),
                    #    )
                    #    print('subcluster loss: ', kl_div)
                    #    print('r_gmm :',r_gmm)
    
                    # Accumulate the KL divergence loss
                    loss += nn.KLDivLoss(reduction="batchmean")(
                        torch.log(r),
                        r_gmm.float().to(device=self.device),
                    )
    
                # Update the subcluster offset
                subcluster_offset += n_sub_k
    
            # **Debugging Step: Compare Parent Cluster Assignments Based on Log-Likelihoods**
            with torch.no_grad():
                # Create a tensor mapping each subcluster to its parent cluster
                # Shape: (total_num_subclusters,)
                subcluster_parent_tensor = torch.tensor(subcluster_parent, device=self.device)
    
                # Compute log-sum-exp of log probabilities for each parent cluster
                # This aggregates log-likelihoods per parent cluster
                # Shape: (batch_size, K)
                parent_log_likelihood = torch.full(
                    (codes.size(0), K), 
                    -float('inf'), 
                    device=self.device
                )
                for k in range(K):
                    # Get indices of subclusters belonging to parent cluster k
                    subcluster_indices = (subcluster_parent_tensor == k).nonzero(as_tuple=True)[0]
                    if len(subcluster_indices) == 0:
                        continue
                    # Use log-sum-exp to aggregate log probabilities
                    parent_log_likelihood[:, k] = torch.logsumexp(all_log_probs[:, subcluster_indices], dim=1)
    
                # Determine the parent cluster with the highest log-likelihood
                inferred_parent = parent_log_likelihood.argmax(dim=1)
    
                # Compare with original parent cluster assignments 'z'
                correct_assignments = (inferred_parent == z).float()
                accuracy = correct_assignments.mean().item()
                """
                if batch_idx == total_batches - 1:
                    print(f"Parent Cluster Assignment Accuracy: {accuracy * 100:.2f}%")
                    mismatches = (inferred_parent != z).nonzero(as_tuple=True)[0]
                    if len(mismatches) > 0:
                        print(f"Mismatched Data Points: {mismatches.tolist()}")
                        for idx in mismatches[:10]:  # Show up to first 10 mismatches
                            print(f"  Data Point {idx}: Original Cluster = {z[idx].item()}, Inferred Cluster = {inferred_parent[idx].item()}")"""
    
            return loss
        elif self.hparams.subcluster_loss == "KL_GMM_2_distord_log_mapping":
            loss = 0.0
            subcluster_offset = 0  # To track subcluster index offsets
    
            z = logits.argmax(-1)  # Original parent cluster assignments
    
            # === Debugging Section ===
            # Verify subcluster masking if masked_subresp is provided
            if masked_subresp is not None and batch_idx is not None and total_batches is not None:
                if batch_idx == total_batches - 1:  # Perform checks only on the last batch
                    print("\n=== Debugging Subcluster Masking ===")
                    for idx in range(codes.size(0)):  # Iterate over each data point in the batch
                        # Get the main cluster assignment for the current data point
                        main_cluster = z[idx].item()
    
                        # Determine the range of subclusters for this main cluster
                        # Use built-in sum() to ensure native Python integers
                        cluster_offset = sum(n_sub_list[:main_cluster])
                        num_subclusters = n_sub_list[main_cluster]
    
                        # Extract selected and non-selected subclusters
                        try:
                            selected_subclusters = masked_subresp[idx, cluster_offset : cluster_offset + num_subclusters]
                        except IndexError:
                            print(f"Data Point {idx}: IndexError - Check cluster_offset and num_subclusters.")
                            continue  # Skip to the next data point
    
                        # Clone the entire row and set selected subclusters to -inf to get non-selected subclusters
                        non_selected_subclusters = masked_subresp[idx].clone()
                        non_selected_subclusters[cluster_offset : cluster_offset + num_subclusters] = -float('inf')
    
                        # Check conditions
                        non_selected_correct = torch.all(non_selected_subclusters == -float('inf'))
                        selected_correct = torch.all(selected_subclusters != -float('inf'))
    
                        # Print debug information based on the checks
                        if not non_selected_correct or not selected_correct:
                            print(f"Data Point {idx}: Incorrect subcluster masking detected.")
                            if not non_selected_correct:
                                print(f"  Non-selected subclusters are not all -inf.")
                            if not selected_correct:
                                print(f"  Selected subclusters contain -inf values.")
                            print(f"  Mask:\n{masked_subresp[idx]}\n")
                        else:
                            print(f"Data Point {idx}: Subcluster masking is correct.")
                    print("=== End of Debugging Subcluster Masking ===\n")
    
            # Calculate the total number of subclusters across all parent clusters
            total_num_subclusters = sum(n_sub_list)
    
            # Initialize a tensor to store log-likelihoods for each data point across all subclusters
            # Shape: (batch_size, total_num_subclusters)
            all_log_probs = torch.full(
                (codes.size(0), total_num_subclusters), 
                -float('inf'), 
                device=self.device
            )
    
            # Initialize a list to map each subcluster to its parent cluster
            # Length: total_num_subclusters
            subcluster_parent = []
    
            # Iterate over each main cluster
            for k in range(K):
                # Get the number of subclusters for cluster k
                n_sub_k = n_sub_list[k]
                codes_k = codes[z == k]
                r = subresp[z == k, subcluster_offset : subcluster_offset + n_sub_k]
                #print('k: ',k)
                #print('subresp ' , subresp.size())
                #print(f'index:{subcluster_offset} to {subcluster_offset+n_sub_k} ')
                #print(r)
                
                if len(codes_k) > 0:
                    r_gmm = []
                    cluster_label = cluster_labels[k] if cluster_labels else f"Cluster {k}"
                    #if batch_idx == total_batches - 1:
                    #    print(f"Processing cluster '{cluster_label}' (MLP index {k})")
                    for k_sub in range(n_sub_k):
                        subcluster_idx = subcluster_offset + k_sub
                        mu = mus_sub[subcluster_idx].double().to(device=self.device)
                        #subcluster_label = subcluster_labels[k][k_sub] if subcluster_labels else f"Subcluster {k_sub}"
                        #if batch_idx == total_batches - 1:
                        #    print(f"  Subcluster '{subcluster_label}' (subcluster index {subcluster_idx})")
    
                        # Existing computations...
                        r_sub = r[:, k_sub]
                        proj_codes = Log_mapping(codes_k.detach().double(), mu_sub, weights=r_sub)  # Shape: (N_k, D)
                        proj_mu = torch.zeros(mu.size()).to(device=self.device)
    
                        # Correctly indexing covs_sub
                        gmm_k = MultivariateNormal(
                            proj_mu, 
                            covs_sub[subcluster_idx].double().to(device=self.device)
                        )
                        prob_k = gmm_k.log_prob(proj_codes)
                        r_gmm.append((prob_k + torch.log(pis_sub[subcluster_idx])).double())
    
                        if batch_idx == total_batches - 1:
                            print(f'pis_sub[{subcluster_idx}] :', pis_sub[subcluster_idx])
    
                        # **Debugging Step: Store log-likelihoods across all subclusters**
                        # Compute log-probability for all data points (not just codes_k)
                        proj_codes_all = Log_mapping(codes.detach().double(), mu)
                        gmm_all = MultivariateNormal(
                            proj_mu, 
                            covs_sub[subcluster_idx].double().to(device=self.device)
                        )
                        prob_all = gmm_all.log_prob(proj_codes_all) + torch.log(pis_sub[subcluster_idx])
                        all_log_probs[:, subcluster_idx] = prob_all
    
                        # Map subcluster to its parent cluster
                        subcluster_parent.append(k)
    
                    # Stack and normalize r_gmm
                    r_gmm = torch.stack(r_gmm).T  # Shape: (batch_size_k, n_sub_k)
                    max_values, _ = r_gmm.max(dim=1, keepdim=True)
                    r_gmm = r_gmm - torch.log(torch.exp(r_gmm - max_values).sum(dim=1, keepdim=True)) - max_values
                    r_gmm = torch.exp(r_gmm)
                    eps = 1e-5
                    r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(dim=1, keepdim=True)
    
                    if batch_idx == total_batches - 1:
                        kl_div = nn.KLDivLoss(reduction="batchmean")(
                            torch.log(r),
                            r_gmm.float().to(device=self.device),
                        )
                        print('subcluster loss: ', kl_div)
    
                    # Accumulate the KL divergence loss
                    loss += nn.KLDivLoss(reduction="batchmean")(
                        torch.log(r),
                        r_gmm.float().to(device=self.device),
                    )
    
                # Update the subcluster offset
                subcluster_offset += n_sub_k
    
            # **Debugging Step: Compare Parent Cluster Assignments Based on Log-Likelihoods**
            with torch.no_grad():
                # Create a tensor mapping each subcluster to its parent cluster
                # Shape: (total_num_subclusters,)
                subcluster_parent_tensor = torch.tensor(subcluster_parent, device=self.device)
    
                # Compute log-sum-exp of log probabilities for each parent cluster
                # This aggregates log-likelihoods per parent cluster
                # Shape: (batch_size, K)
                parent_log_likelihood = torch.full(
                    (codes.size(0), K), 
                    -float('inf'), 
                    device=self.device
                )
                for k in range(K):
                    # Get indices of subclusters belonging to parent cluster k
                    subcluster_indices = (subcluster_parent_tensor == k).nonzero(as_tuple=True)[0]
                    if len(subcluster_indices) == 0:
                        continue
                    # Use log-sum-exp to aggregate log probabilities
                    parent_log_likelihood[:, k] = torch.logsumexp(all_log_probs[:, subcluster_indices], dim=1)
    
                # Determine the parent cluster with the highest log-likelihood
                inferred_parent = parent_log_likelihood.argmax(dim=1)
    
                # Compare with original parent cluster assignments 'z'
                correct_assignments = (inferred_parent == z).float()
                accuracy = correct_assignments.mean().item()
    
                if batch_idx == total_batches - 1:
                    print(f"Parent Cluster Assignment Accuracy: {accuracy * 100:.2f}%")
                    mismatches = (inferred_parent != z).nonzero(as_tuple=True)[0]
                    if len(mismatches) > 0:
                        print(f"Mismatched Data Points: {mismatches.tolist()}")
                        for idx in mismatches[:10]:  # Show up to first 10 mismatches
                            print(f"  Data Point {idx}: Original Cluster = {z[idx].item()}, Inferred Cluster = {inferred_parent[idx].item()}")
    
            return loss



    def subcluster_loss_function_new_Adecoch2(self, codes, logits, subresp, K, n_sub_list, mus_sub, covs_sub=None, pis_sub=None,cluster_labels=None, subcluster_labels=None, batch_idx=None, total_batches=None):
        if self.hparams.subcluster_loss == "cosine_dissimilarity":
            # Implement cosine dissimilarity loss if applicable
            pass
        elif self.hparams.cluster_loss == "KL_GMM_2":
            loss = 0
            subcluster_offset = 0  # To track subcluster index offsets
    
            z = logits.argmax(-1)
            for k in range(K):
                # Get the number of subclusters for cluster k
                n_sub_k = n_sub_list[k]
                codes_k = codes[z == k]
                r = subresp[z == k, subcluster_offset: subcluster_offset + n_sub_k]
    
                if len(codes_k) > 0:
                    r_gmm = []
                    cluster_label = cluster_labels[k] if cluster_labels else f"Cluster {k}"
                    if batch_idx == total_batches - 1:
                        print(f"Processing cluster '{cluster_label}' (MLP index {k})")
                    for k_sub in range(n_sub_k):
                        subcluster_idx = subcluster_offset + k_sub
                        mu = mus_sub[subcluster_idx].double().to(device=self.device)
                        subcluster_label = subcluster_labels[k][k_sub] if subcluster_labels else f"Subcluster {k_sub}"
                        if batch_idx == total_batches - 1:
                            print(f"  Subcluster '{subcluster_label}' (subcluster index {subcluster_idx})")
                        # Existing computations...
                        proj_codes = Log_mapping(codes_k.detach().double(), mu)
                        proj_mu = torch.zeros(mu.size()).to(device=self.device)
                        gmm_k = MultivariateNormal(proj_mu, covs_sub[subcluster_idx].double().to(device=self.device))
                        prob_k = gmm_k.log_prob(proj_codes)
                        r_gmm.append((prob_k + torch.log(pis_sub[subcluster_idx])).double())
                        if batch_idx==total_batches-1:
                          print(f'pis_sub[{subcluster_idx}] :',pis_sub[subcluster_idx])
                    r_gmm = torch.stack(r_gmm).T
                    max_values, _ = r_gmm.max(axis=1, keepdim=True)
                    r_gmm -= torch.log(torch.exp(r_gmm - max_values).sum(axis=1, keepdim=True)) + max_values
                    r_gmm = torch.exp(r_gmm)
                    eps = 1e-5
                    r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1, keepdim=True)
                    r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)
                    if batch_idx == total_batches - 1:
                        print('subcluster loss: ',nn.KLDivLoss(reduction="batchmean")(
                          torch.log(r),
                          r_gmm.float().to(device=self.device),))
                    loss += nn.KLDivLoss(reduction="batchmean")(
                        torch.log(r),
                        r_gmm.float().to(device=self.device),
                    )
                subcluster_offset += n_sub_k
            return loss

            
    def subcluster_loss_function_new_Adecoch(self, codes, logits, subresp, K, n_sub_list, mus_sub, covs_sub=None, pis_sub=None,is_split=False):
        if self.hparams.subcluster_loss == "cosine_dissimilarity":
            pass
    
        elif self.hparams.cluster_loss == "KL_GMM_2":
            loss = 0
            subcluster_offset = 0  # To track subcluster index offsets
    
            for k in range(K):
                # Get the number of subclusters for cluster k
                n_sub_k = n_sub_list[k]
    
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                r = subresp[z == k, subcluster_offset: subcluster_offset + n_sub_k]
                if is_split :
                  print('subresp :', subresp.size())
                  print('Cluster Index:', k)
                  print('Number of subclusters for cluster {}: {}'.format(k, n_sub_k))
                  print('R KL GMM 2:', r.size())
                  print('R :', r)
                  print('Codes_k size:', codes_k.size())
                  print('Codes_k:', codes_k)
    
                if len(codes_k) > 0:
                    r_gmm = []
                    for k_sub in range(n_sub_k):
                        mu = mus_sub[subcluster_offset + k_sub].double().to(device=self.device)
                        proj_codes = Log_mapping(codes_k.detach().double(), mu)
                        proj_mu = torch.zeros(mu.size()).to(device=self.device)
    
                        gmm_k = MultivariateNormal(proj_mu, covs_sub[subcluster_offset + k_sub].double().to(device=self.device))
                        prob_k = gmm_k.log_prob(proj_codes)
                        
                        if is_split:
                          print(f"Subcluster {k_sub} | mu:", mu)
                          print(f"Subcluster {k_sub} | proj_codes:", proj_codes)
                          print(f"Subcluster {k_sub} | prob_k:", prob_k)
    
                        r_gmm.append((prob_k + torch.log(pis_sub[subcluster_offset + k_sub])).double())
    
                    r_gmm = torch.stack(r_gmm).T
                    
    
                    max_values, _ = r_gmm.max(axis=1, keepdim=True)
                    r_gmm -= torch.log(torch.exp(r_gmm - max_values).sum(axis=1, keepdim=True)) + max_values
                    r_gmm = torch.exp(r_gmm)
    
                    
    
                    eps = 0.00001
                    r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1, keepdim=True)
                    r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)
    
                    #print('R (after normalization):', r)
                    #print('R_gmm (after normalization):', r_gmm)
    
                    loss_component = nn.KLDivLoss(reduction="batchmean")(
                        torch.log(r),
                        r_gmm.float().to(device=self.device),
                    )
                    if is_split:
                      print(f"R_gmm (before normalization): {r_gmm}")
                      print('KL Loss component:', loss_component)
                      print(f"R_gmm (after logsumexp normalization): {r_gmm}")
                      print('Final LOSS:', loss)
    
                    loss += loss_component
    
                # Update the subcluster offset
                subcluster_offset += n_sub_k
    
            #
            return loss

    
        raise NotImplementedError("No such loss!")

    def subcluster_loss_function_new_DPM(
        self, codes, logits, subresp, K, n_sub, mus_sub, covs_sub=None, pis_sub=None
    ):
        
        if self.hparams.subcluster_loss == "cosine_dissimilarity":
           
            codes_norm = F.normalize(codes, p=2, dim=-1)
            mus_sub_norm = F.normalize(mus_sub.to(device=self.device), p=2, dim=-1)
            #print("Does codes_norm have gradients?", codes_norm.requires_grad)
            #print("Does mus_sub_norm have gradients?", mus_sub_norm.requires_grad)
            # Expand dimensions to allow broadcasting and comparison of each code with each mus_sub
            C_tag = codes_norm.unsqueeze(1).expand(-1, 2 * K, -1)
            mus_tag = mus_sub_norm.unsqueeze(0).expand(codes.shape[0], -1, -1)
            #print("Does mus_tag have gradients?", mus_tag.requires_grad)
            #print("Does subresp have gradients?", subresp.requires_grad)
            # Compute cosine similarity for each pair (code, mus_sub)
            # Using 1 - cosine_similarity to treat it as a loss (distance measure)
            cosine_dist = 1 - F.cosine_similarity(C_tag, mus_tag, dim=2)
            
            # Flatten subresp to match the flattened structure of C_tag and mus_tag distances
            r_tag = subresp.flatten()
            #print('r_tag',r_tag.size())
            #print('r_tag require grad',r_tag.requires_grad)
            # Weight the cosine distance by the subcluster responsibilities and average
            loss = (r_tag * cosine_dist.flatten()).sum()/len(codes)
            #print('does loss requires.grad',loss.requires_grad)
            return loss
        
        elif self.hparams.cluster_loss == "KL_GMM_2":
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                r = subresp[z == k, 2 * k: 2 * k + 2]
                if len(codes_k) > 0:
                    r_gmm = []
                    for k_sub in range(n_sub):
                        mu=mus_sub[2 * k + k_sub].double().to(device=self.device)
                        proj_codes=Log_mapping(codes_k.detach().double(), mu)
                        proj_mu=torch.zeros(mu.size()).to(device=self.device)
                        #print("COVS_SUB",covs_sub)
                        gmm_k = MultivariateNormal(proj_mu, covs_sub[2 * k + k_sub].double().to(device=self.device))
                        prob_k = gmm_k.log_prob(proj_codes)
                        r_gmm.append((prob_k + torch.log(pis_sub[2 * k + k_sub])).double())
                    r_gmm = torch.stack(r_gmm).T
                    max_values, _ = r_gmm.max(axis=1, keepdim=True)
                    r_gmm -= torch.log(torch.exp((r_gmm - max_values)).sum(axis=1, keepdim=True)) + max_values
                    r_gmm = torch.exp(r_gmm)
                    eps = 0.00001
                    r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1, keepdim=True)
                    r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)
                    loss += nn.KLDivLoss(reduction="batchmean")(
                            torch.log(r),
                            r_gmm.float().to(device=self.device),
                    )
            return loss

        elif self.hparams.subcluster_loss == "diag_NIG":
            # NIG
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                if codes_k.shape[0] > 0:
                    # else: handle empty clusters, won't necessarily happen
                    for k_sub in range(n_sub):
                        r = subresp[z == k, k, :][:, 2 * k + k_sub]
                        mus_tag = mus_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        sigma_sub = torch.sqrt(
                            covs_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        )
                        loss += (
                            r
                            * (
                                (
                                    torch.norm(
                                        (codes_k - mus_tag.to(device=self.device) / sigma_sub.to(device=self.device)),
                                        dim=1,
                                    )
                                )
                                ** 2
                            )
                        ).sum()
            return loss

        raise NotImplementedError("No such loss!")
            
    def cluster_loss_function_DPM(
        self, c, r, model_mus, K, codes_dim, model_covs=None, pi=None, logger=None, 
    ):
        if self.hparams.cluster_loss == "isotropic":
            # Isotropic
            C_tag = c.repeat(1, K).view(-1, codes_dim)
            mus_tag = model_mus.repeat(c.shape[0], 1)
            r_tag = r.flatten()
            return (r_tag * ((torch.norm(C_tag - mus_tag.to(device=self.device), dim=1)) ** 2)).mean()

        elif self.hparams.cluster_loss == "diag_NIG":
            # NIG prior
            # K * N, D
            C_tag = c.repeat(1, K).view(-1, codes_dim)
            sigmas = torch.sqrt(model_covs).repeat(c.shape[0], 1)
            mus_tag = model_mus.repeat(c.shape[0], 1)
            r_tag = r.flatten()
            return (
                r_tag
                * ((torch.norm((C_tag - mus_tag.to(device=self.device)) / sigmas.to(device=self.device), dim=1)) ** 2)
            ).mean()

        elif self.hparams.cluster_loss == "KL_GMM_2":
            r_gmm = []
            for k in range(K):
                gmm_k = MultivariateNormal(model_mus[k].double().to(device=self.device), model_covs[k].double().to(device=self.device))
                prob_k = gmm_k.log_prob(c.detach().double())
                r_gmm.append((prob_k + torch.log(pi[k])).double())
            r_gmm = torch.stack(r_gmm).T
            max_values, _ = r_gmm.max(axis=1, keepdim=True)
            r_gmm -= torch.log(torch.exp((r_gmm - max_values)).sum(axis=1, keepdim=True)) + max_values
            r_gmm = torch.exp(r_gmm)
            eps = 0.00001
            r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1, keepdim=True)
            r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)
            
            return nn.KLDivLoss(reduction="batchmean")(
                torch.log(r),
                r_gmm.float().to(device=self.device),
            )

        raise NotImplementedError("No such loss")

    def subcluster_loss_function(
        self, codes, logits, subresp, K, n_sub, mus_sub, covs_sub=None, pis_sub=None
    ):
        if self.hparams.subcluster_loss == "isotropic":
            # Isotropic
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]

                if codes_k.shape[0] > 0:
                    # else: handle empty clusters, won't necessarily happen
                    for k_sub in range(n_sub):
                        r = subresp[z == k, 2 * k + k_sub]
                        mus_tag = mus_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        loss += (
                            r * ((torch.norm(codes_k - mus_tag.to(device=self.device), dim=1)) ** 2)
                        ).sum()
            return loss / float(len(codes))

        elif self.hparams.cluster_loss == "KL_GMM_2":
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                r = subresp[z == k, 2 * k: 2 * k + 2]
                if len(codes_k) > 0:
                    r_gmm = []
                    for k_sub in range(n_sub):
                        gmm_k = MultivariateNormal(mus_sub[2 * k + k_sub].double().to(device=self.device), covs_sub[2 * k + k_sub].double().to(device=self.device))
                        prob_k = gmm_k.log_prob(codes_k.detach().double())
                        r_gmm.append((prob_k + torch.log(pis_sub[2 * k + k_sub])).double())
                    r_gmm = torch.stack(r_gmm).T
                    max_values, _ = r_gmm.max(axis=1, keepdim=True)
                    r_gmm -= torch.log(torch.exp((r_gmm - max_values)).sum(axis=1, keepdim=True)) + max_values
                    r_gmm = torch.exp(r_gmm)
                    eps = 0.00001
                    r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1, keepdim=True)
                    r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)
                    loss += nn.KLDivLoss(reduction="batchmean")(
                            torch.log(r),
                            r_gmm.float().to(device=self.device),
                    )
            return loss

        elif self.hparams.subcluster_loss == "diag_NIG":
            # NIG
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                if codes_k.shape[0] > 0:
                    # else: handle empty clusters, won't necessarily happen
                    for k_sub in range(n_sub):
                        r = subresp[z == k, k, :][:, 2 * k + k_sub]
                        mus_tag = mus_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        sigma_sub = torch.sqrt(
                            covs_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        )
                        loss += (
                            r
                            * (
                                (
                                    torch.norm(
                                        (codes_k - mus_tag.to(device=self.device) / sigma_sub.to(device=self.device)),
                                        dim=1,
                                    )
                                )
                                ** 2
                            )
                        ).sum()
            return loss

        raise NotImplementedError("No such loss!")

    def subcluster_loss_function_new_DPM(
        self, codes, logits, subresp, K, n_sub, mus_sub, covs_sub=None, pis_sub=None
    ):
        if self.hparams.subcluster_loss == "isotropic":
            # Isotropic

            C_tag = codes.repeat(1, 2 * K).view(-1, codes.size(1))
            mus_tag = mus_sub.repeat(codes.shape[0], 1)
            r_tag = subresp.flatten()
            
            return (r_tag * ((torch.norm(C_tag - mus_tag.to(device=self.device), dim=1)) ** 2)).sum() / float(len(codes))

        elif self.hparams.cluster_loss == "KL_GMM_2":
            loss = 0
            for k in range(K):
                print("Does mus_sub have gradients?", mus_sub.requires_grad)
                print("Does subresp have gradients?", subresp.requires_grad)
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                r = subresp[z == k, 2 * k: 2 * k + 2]
                if len(codes_k) > 0:
                    r_gmm = []
                    for k_sub in range(n_sub):
                        gmm_k = MultivariateNormal(mus_sub[2 * k + k_sub].double().to(device=self.device), covs_sub[2 * k + k_sub].double().to(device=self.device))
                        prob_k = gmm_k.log_prob(codes_k.detach().double())
                        r_gmm.append((prob_k + torch.log(pis_sub[2 * k + k_sub])).double())
                    r_gmm = torch.stack(r_gmm).T
                    max_values, _ = r_gmm.max(axis=1, keepdim=True)
                    r_gmm -= torch.log(torch.exp((r_gmm - max_values)).sum(axis=1, keepdim=True)) + max_values
                    r_gmm = torch.exp(r_gmm)
                    eps = 0.00001
                    r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1, keepdim=True)
                    r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)
                    loss += nn.KLDivLoss(reduction="batchmean")(
                            torch.log(r),
                            r_gmm.float().to(device=self.device),
                    )
            return loss

        elif self.hparams.subcluster_loss == "diag_NIG":
            # NIG
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                if codes_k.shape[0] > 0:
                    # else: handle empty clusters, won't necessarily happen
                    for k_sub in range(n_sub):
                        r = subresp[z == k, k, :][:, 2 * k + k_sub]
                        mus_tag = mus_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        sigma_sub = torch.sqrt(
                            covs_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        )
                        loss += (
                            r
                            * (
                                (
                                    torch.norm(
                                        (codes_k - mus_tag.to(device=self.device) / sigma_sub.to(device=self.device)),
                                        dim=1,
                                    )
                                )
                                ** 2
                            )
                        ).sum()
            return loss

        raise NotImplementedError("No such loss!")

    def comp_std(self, codes, hard_assignments, K):
        stds = []
        for k in range(K):
            codes_k = codes[hard_assignments == k]
            if len(codes_k > 0):
                per_dim_std = codes_k.std(axis=0)
            else:
                per_dim_std = torch.sqrt(codes.std(axis=0))
            stds.append(per_dim_std)
        return torch.stack(stds)

    def autoencoder_kl_dist_loss_function(
        z, mu, log_var, hard_assign, model_mus, model_std, mean=False
    ):
        z = z.detach()
        p = torch.distributions.Normal(model_mus[hard_assign], model_std[hard_assign])
        log_prob_p_z = p.log_prob(z)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        log_prob_q_z = q.log_prob(z)

        dist_kl = log_prob_p_z - log_prob_q_z
        dist_kl *= log_prob_p_z.exp()
        if mean:
            dist_kl = dist_kl.mean()
        else:
            dist_kl = dist_kl.sum()
        return dist_kl

    def update_labels_after_split_merge(
        self,
        hard_assign,
        split_performed,
        merge_performed,
        mus,
        mus_ind_to_split,
        mus_inds_to_merge,
        resp_sub,
    ):
        cluster_net_labels = hard_assign
        if split_performed or merge_performed:
            if split_performed:
                label_map = {}
                count = 0
                count_split = 0
                second_subcluster_inds = torch.tensor([])
                for mu_ind in range(len(mus)):
                    if mu_ind in mus_ind_to_split:
                        mask_current_mu = cluster_net_labels == mu_ind
                        # first cluster
                        label_map[mu_ind] = (
                            len(mus) - len(mus_ind_to_split) + count_split
                        )
                        # list second_subcluster so we will remember to increase its label by one
                        sub_assign = resp_sub[
                            mask_current_mu, mu_ind, 2 * mu_ind: 2 * mu_ind + 2
                        ].argmax(-1)
                        inds_current_mu = mask_current_mu.nonzero(as_tuple=False)
                        second_subcluster_inds = torch.cat(
                            [second_subcluster_inds, inds_current_mu[sub_assign == 1]]
                        )
                        count_split += 2
                    else:
                        label_map[mu_ind] = count
                        count += 1
                new_labels = torch.zeros_like(cluster_net_labels) - 1
                for key, value in label_map.items():
                    new_labels[cluster_net_labels == key] = value
                new_labels[
                    second_subcluster_inds.clone().detach().type(torch.long)
                ] += 1
            elif merge_performed:
                count = 0
                label_map = {}
                pairs = torch.zeros(len(mus_inds_to_merge))

                for mu_ind in range(len(mus)):
                    if mu_ind in mus_inds_to_merge.flatten():
                        which_pair = (mus_inds_to_merge == mu_ind).nonzero(
                            as_tuple=False
                        )[0][0]
                        if pairs[which_pair] == 0:
                            # first, open new cluster
                            label_map[mu_ind] = (
                                len(mus) - len(mus_inds_to_merge.flatten()) + which_pair
                            )
                            pairs[which_pair] += 1
                        else:
                            # second, join the already opened cluster
                            # find the first of this pair
                            which_pair_col = (mus_inds_to_merge == mu_ind).nonzero(
                                as_tuple=False
                            )[0][1]
                            first = mus_inds_to_merge[
                                which_pair, (which_pair_col + 1) % 2
                            ]
                            label_map[mu_ind] = label_map[first.item()]
                    else:
                        label_map[mu_ind] = count
                        count += 1
                new_labels = torch.zeros_like(cluster_net_labels)
                for key, value in label_map.items():
                    new_labels[cluster_net_labels == key] = value
            return new_labels

    def should_init_em(self, split_performed, merge_performed, previous_training_stage, current_stage):
        # A flag to reinitialized the EM object. Should occur in two occasions:
        # 1. The embeddings have changed
        # 2. K has changes
        K_changed = split_performed or merge_performed
        embeddings_changed = previous_training_stage in ["pretrain_ae", 'only_ae', "train_ae_w_add_loss", "only_ae_w_cluster_loss", "train_together", ]
        if (K_changed or embeddings_changed) and current_stage == "only_cluster_net":
            return True
        return False

    def should_perform_em(
        self, current_epoch, split_performed, merge_performed, previous_training_stage, current_stage
    ):
        return current_epoch > 0 and (
            self.should_init_em(
                split_performed, merge_performed, previous_training_stage, current_stage
            )
            and self.hparams.cluster_loss == "KL_GMM"
        )

    @staticmethod
    def update_following_split(mus, mus_ind_to_split, train_resp_sub, cluster_net_labels):
        label_map = {}
        count = 0
        count_split = 0
        second_subcluster_inds = torch.tensor([])
        for mu_ind in range(len(mus)):
            if mu_ind in mus_ind_to_split:
                mask_current_mu = cluster_net_labels == mu_ind
                # first cluster
                label_map[mu_ind] = (
                    len(mus) - len(mus_ind_to_split) + count_split
                )
                # list second_subcluster so we will remember to increase its label by one
                sub_assign = train_resp_sub[
                    mask_current_mu, mu_ind, 2 * mu_ind: 2 * mu_ind + 2
                ].argmax(-1)
                inds_current_mu = mask_current_mu.nonzero(as_tuple=False)
                second_subcluster_inds = torch.cat(
                    [second_subcluster_inds, inds_current_mu[sub_assign == 1]]
                )
                count_split += 2

            else:
                label_map[mu_ind] = count
                count += 1
        new_labels = torch.zeros_like(cluster_net_labels) - 1
        for key, value in label_map.items():
            new_labels[cluster_net_labels == key] = value
        new_labels[
            second_subcluster_inds.clone().detach().type(torch.long)
        ] += 1
        return new_labels

    @staticmethod
    def update_following_merge(mus, mus_inds_to_merge, cluster_net_labels):
        count = 0
        label_map = {}
        pairs = torch.zeros(len(mus_inds_to_merge))
        for mu_ind in range(len(mus)):
            if mu_ind in mus_inds_to_merge.flatten():
                which_pair = (mus_inds_to_merge == mu_ind).nonzero(
                    as_tuple=False
                )[0][0]
                if pairs[which_pair] == 0:
                    # first, open new cluster
                    label_map[mu_ind] = (
                        len(mus)
                        - len(mus_inds_to_merge.flatten())
                        + which_pair
                    )
                    pairs[which_pair] += 1
                else:
                    # second, join the already opened cluster
                    # find the first of this pair
                    which_pair_col = (mus_inds_to_merge == mu_ind).nonzero(
                        as_tuple=False
                    )[0][1]
                    first = mus_inds_to_merge[
                        which_pair, (which_pair_col + 1) % 2
                    ]
                    label_map[mu_ind] = label_map[first.item()]
            else:
                label_map[mu_ind] = count
                count += 1
        new_labels = torch.zeros_like(cluster_net_labels)
        for key, value in label_map.items():
            new_labels[cluster_net_labels == key] = value
        return new_labels

    def log_metric(self, metric_name, metric_val):
        self.log(metric_name, metric_val)

    @staticmethod
    def get_updated_net_labels(cluster_net_labels, split_performed, merge_performed, mus, mus_ind_to_split, mus_inds_to_merge, train_resp_sub):
        """ Compute the updated net labels if a split/merge has occured in this epoch
        """
        if split_performed:
            return training_utils.update_following_split(mus, mus_ind_to_split, train_resp_sub, cluster_net_labels)
        elif merge_performed:
            return training_utils.update_following_merge(mus, mus_inds_to_merge, cluster_net_labels)
    
    #Assigne un label par cluster en fct du nombre d'occurent des gt labels . CHAQUE LABELS EST ASSGINE QU'une fois 
    @staticmethod
    def _best_cluster_fit(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        row_ind, col_ind = linear_assignment(w.max() - w)
        map_dict = {}
        for j in range(len(col_ind)):
            map_dict[col_ind[j]] = row_ind[j]
        y_true_new = np.array([map_dict[i] for i in y_true])
        return y_true_new, row_ind, col_ind, w

    @staticmethod
    def cluster_acc(y_true, y_pred, y_pred_top5=None):
        y_true_new, row_ind, col_ind, w = training_utils._best_cluster_fit(y_true.numpy(), y_pred.numpy())
        if y_pred_top5 is not None:
            y_true_new = torch.from_numpy(y_true_new).unsqueeze(0).repeat(5, 1)
            acc_top5 = (y_pred_top5.T == y_true_new).any(axis=0).sum() * 1.0 / y_pred.numpy().size
            acc_top5 = acc_top5.item()
        else:
            acc_top5 = 0.
        
        return acc_top5, np.round(w[row_ind, col_ind].sum() * 1.0 / y_pred.numpy().size, 5)

    def save_cluster_examples(self, logits, x_for_vis, y, epoch, init_num=0, num_img=20, grid_size=8):
        # save 20 examples of each cluster and also record its true label class.
        K = logits.shape[1]
        hard_assign = logits.argmax(-1)
        for k in range(K):
            # take images of the cluster
            x_k = x_for_vis[hard_assign == k][:num_img]
            y_gt = y[hard_assign == k][:num_img]
            # save each image
            for i in range(min(num_img, x_k.shape[0])):
                save_image(x_k[i], f"{self.hparams.dataset}_imgs/clusternet{init_num}_epoch{epoch}_clus{k}_label{y_gt[i]}_{i}.jpeg")
            # save as a grid
            num_imgs = min(grid_size, x_k.shape[0])
            if num_imgs > 0:
                grid = make_grid(x_k[:num_imgs], nrow=num_imgs)
                save_image(grid, f"{self.hparams.dataset}_imgs/clusternet{init_num}_epoch{epoch}_clus{k}.jpeg")

    def save_batch_of_images(self, x_for_vis, nrow=8, rows=5):
        x = x_for_vis[:rows * nrow]
        grid = make_grid(x, nrow=nrow)
        save_image(grid, f"{self.hparams.dataset}_imgs/{self.hparams.dataset}_grid.jpeg")
