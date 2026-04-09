#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
from torch import mvlgamma
from torch import lgamma
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.special import multigammaln ,gammaln
import math
import torch.nn.functional as F
from src.clustering_models.clusternet_modules.utils.clustering_utils.clustering_operations import Log_mapping,Exp_mapping,KarcherMean


class Priors:
    '''
    A prior that will hold the priors for all the parameters.
    '''
    def __init__(self, hparams, K, codes_dim, counts=10, prior_sigma_scale=None, prior_choice='data_std'):
        self.name = "prior_class"
        self.pi_prior_type = hparams.pi_prior
        self.prior_choice=prior_choice
        if hparams.pi_prior:
            self.pi_prior = Dirichlet_prior(K, hparams.pi_prior, counts)
        else:
            self.pi_prior = None
        if hparams.prior == "NIW":
            self.mus_covs_prior = NIW_prior(hparams, prior_sigma_scale,)
        #elif hparams.prior == "NIG":
        #    self.mus_covs_prior = NIG_prior(hparams, codes_dim)
        self.name = self.mus_covs_prior.name
        self.pi_counts = hparams.prior_dir_counts
        self.use_prior=hparams.use_priors
        
        

    def update_pi_prior(self, K_new, counts=10, pi_prior=None):
        # pi_prior = None- keep the same pi_prior type
        if self.pi_prior:
            if pi_prior:
                self.pi_prioir = Dirichlet_prior(K_new, pi_prior, counts)
            self.pi_prior = Dirichlet_prior(K_new, self.pi_prior_type, counts)

    def comp_post_counts(self, counts):
        if self.pi_prior:
            return self.pi_prior.comp_post_counts(counts)
        else:
            return counts

    def comp_post_pi(self, pi):
        if self.pi_prior:
            return self.pi_prior.comp_post_pi(pi, self.pi_counts)
        else:
            return pi

    def get_sum_counts(self):
        return self.pi_prior.get_sum_counts()
    
    def init_priors(self, codes,labels=None):
        return self.mus_covs_prior.init_priors(codes,labels=None)
        
    #def init_priors(self, codes):
    #    return self.mus_covs_prior.init_priors(codes)

    def compute_params_post(self, codes_k, mu_k):
        return self.mus_covs_prior.compute_params_post(codes_k, mu_k)

    def compute_post_mus(self, N_ks, data_mus):
        return self.mus_covs_prior.compute_post_mus(N_ks, data_mus)

    def compute_post_cov(self, N_k, mu_k, data_cov_k,psi_index=None,psi_value=None,log_scaling=False):
        return self.mus_covs_prior.compute_post_cov(N_k, mu_k, data_cov_k,psi_index,psi_value=psi_value,log_scaling=log_scaling)

    def log_marginal_likelihood(self, codes_k, covs_k):
        return self.mus_covs_prior.log_marginal_likelihood(codes_k, covs_k)
    
    def log_marginal_likelihood_dynamic_psi(self,codes_k,sigmak,psi_index=None,psi_value=None):
        return self.mus_covs_prior.log_marginal_likelihood_dynamic_psi(codes_k,sigmak,psi_index,psi_value)
    def log_mean_projected_data(self,codes,Sigma_k):
        return self.mus_covs_prior.log_mean_projected_data(codes,Sigma_k)    
    
    def log_mean_projected_data_previous(self,codes, mu_k, Sigma_k):
        return self.mus_covs_prior.log_mean_projected_data(codes, mu_k, Sigma_k)
    
    def get_priors(self,k):
        return self.mus_covs_prior.niw_psi_clusters[k]
    def get_prior_sigma_scale(self):
        return self.mus_covs_prior.prior_sigma_scale
    

class Dirichlet_prior:
    def __init__(self, K, pi_prior="uniform", counts=10):
        self.name = "Dirichlet_dist"
        self.K = K
        self.counts = counts
        if pi_prior == "uniform":
            self.p_counts = torch.ones(K) * counts
            self.pi = self.p_counts / float(K * counts)

    def comp_post_counts(self, counts=None):
        if counts is None:
            counts = self.counts
        return counts + self.p_counts

    def comp_post_pi(self, pi, counts=None):
        if counts is None:
            # counts = 0.001
            counts = 0.1
        return (pi + counts) / (pi + counts).sum()

    def get_sum_counts(self):
        return self.K * self.counts


class NIW_prior:
    """A class used to store niw parameters and compute posteriors.
    Used as a class in case we will want to update these parameters.
    """
    
    def __init__(self, hparams, prior_sigma_scale=None):
        self.name = "NIW"
        self.prior_mu_0_choice = hparams.prior_mu_0
        self.prior_sigma_choice = hparams.prior_sigma_choice
        self.prior_sigma_scale = prior_sigma_scale or hparams.prior_sigma_scale
        self.niw_kappa = hparams.prior_kappa
        self.niw_nu = hparams.NIW_prior_nu
        #print('IW NU :',self.niw_nu)
    
    

    def compute_spherical_data_spread(self,codes, n):
        """
        Compute the data spread for a dataset distributed on a unit sphere.
        
        Parameters:
        - codes: Tensor of shape (m, d) where m is the number of points and d is the dimensionality of the sphere.
        - n: Number of random points to select for the log mapping.
    
        Returns:
        - avg_std_matrix: Tensor of shape (d,) containing the average standard deviation for each dimension.
        """
        m, d = codes.shape
        
        # Randomly select n base points (mu) from the dataset
        indices = torch.randint(0, m, (n,))
        base_points = codes[indices]  # Shape (n, d)
        
        std_matrices = []
        
        for i in range(n):
            mu = base_points[i]  # Shape (d,)
            
            # Apply log mapping
            log_mapped_codes = Log_mapping(codes, mu)  # Assuming Log_mapping outputs shape (m, d)
            
            # Compute standard deviation across all points for each dimension
            std_matrix = log_mapped_codes.std(dim=0)  # Shape (d,)
            std_matrices.append(std_matrix)
        
        # Stack all std matrices and compute the average
        std_matrices = torch.stack(std_matrices)  # Shape (n, d)
        avg_std_matrix = std_matrices.mean(dim=0)  # Shape (d,)
        
        return avg_std_matrix

    def gt_cluster_std(self,codes,gt):
        clusters=[ codes[torch.where(gt==i)] for i in range(len(torch.unique(gt)))]
        cluster_centers=[KarcherMean(None,codes=clusters[i]) for i in range(len(clusters))]
        proj_clusters=[Log_mapping(clusters[i],cluster_centers[i]) for i in range(len(clusters))] 
        std_mean_clusters=[torch.std(proj_clusters[i],dim=0) for i in range(len(proj_clusters))]
        mean_clusters_std=torch.mean(torch.stack(std_mean_clusters,dim=0),dim=0)
        return mean_clusters_std

    def init_priors(self,codes,labels=None):
        if self.prior_mu_0_choice == "data_mean":
            self.niw_m = codes.mean(axis=0)
        if self.prior_sigma_choice == "isotropic":
            self.niw_psi = (torch.eye(codes.shape[1]) * self.prior_sigma_scale).double()
        elif self.prior_sigma_choice == "data_std":
            #self.niw_psi = (torch.diag(codes.std(axis=0)) * self.prior_sigma_scale).double()
            mu=KarcherMean(None,codes)
            proj_codes=Log_mapping(codes,mu)
            self.niw_psi =(torch.diag(proj_codes.std(axis=0))*self.prior_sigma_scale).double()
            #self.niw_psi =(torch.cov(proj_codes)*self.prior_sigma_scale).double()
            #self.niw_psi = torch.zeros((codes.size()[1],codes.size()[1])).double()
            #self.niw_psi_scaled=(1.0**2)*self.niw_psi
            #print('IW_PSI :',self.niw_psi)
        elif self.prior_sigma_choice == "data_std_v2":
            self.niw_psi =(torch.diag(self.compute_spherical_data_spread(codes, n=100))*self.prior_sigma_scale).double()
            print('IW_PSI :',self.niw_psi)
        #elif self.prior_sigma_choice == "gt_data_std":
             #self.niw_psi=(torch.diag(self.gt_cluster_std(codes,labels)*self.prior_sigma_scale).double())
             #print('IW_PSI :',self.niw_psi)
        elif self.prior_sigma_choice == "dynamic_data_std":
             index_labels=torch.unique(labels)
             print('linit prior abels K :',index_labels)
             self.niw_psi_clusters = [torch.unsqueeze((torch.diag(Log_mapping(codes[torch.where(labels==k)],KarcherMean(None,codes[torch.where(labels==k)])).std(axis=0))*self.prior_sigma_scale),0).double() for k in index_labels]
             self.niw_psi_clusters = torch.cat(self.niw_psi_clusters)
             print('NIW_PSI CLUSTERS :', self.niw_psi_clusters.size())
             
        else:
            raise NotImplementedError()
        if self.prior_sigma_choice=='dynamic_data_std' :
           return self.niw_m, self.niw_psi_clusters
        else :
        
          return self.niw_m, self.niw_psi

    def compute_params_post(self, codes_k, mu_k):
        # This is in HARD assignment.
        N_k = len(codes_k)
        sum_k = codes_k.sum(axis=0)
        kappa_star = self.niw_kappa + N_k
        nu_star = self.niw_nu + N_k
        mu_0_star = (self.niw_m * self.niw_kappa + sum_k) / kappa_star
        codes_minus_mu = codes_k - mu_k
        S = codes_minus_mu.T @ codes_minus_mu
        psi_star = (
            self.niw_psi
            + S
            + (self.niw_kappa * N_k / kappa_star)
            * (mu_k - self.niw_m).unsqueeze(1)
            @ (mu_k - self.niw_m).unsqueeze(0)
        )
        return kappa_star, nu_star, mu_0_star, psi_star

    def compute_post_mus(self, N_ks, data_mus):
        # N_k is the number of points in cluster K for hard assignment, and the sum of all responses to the K-th cluster for soft assignment
        #print("N_ks : ",N_ks)
        return ((N_ks.reshape(-1, 1) * data_mus) + (self.niw_kappa * self.niw_m)) / (
            N_ks.reshape(-1, 1) + self.niw_kappa
        )
    
    def update_update_niw_psi(new_value,k):
       self.niw_psi_clusters[k]=new_value
    def append_new_niw_psi(new_value):
       self.niw_psi_clusters.append(new_value)
    def compute_post_cov(self, N_k, data_cov_k, D, psi_index=None, psi_value=None, log_scaling=False):
        # If it is hard assignments: N_k is the number of points assigned to cluster K, x_mean is their average
        # If it is soft assignments: N_k is the r_k, the sum of responses to the k-th cluster, x_mean is the data average (all the data)
        
        print("N_k", N_k)
        print("N_k size", N_k.size())
    
        # Determine the psi to use
        if psi_value is not None:
            psi = psi_value.to(data_cov_k.device)
        elif psi_index is not None:
            psi = self.niw_psi_clusters[psi_index].to(data_cov_k.device)
        else:
            psi = self.niw_psi.to(data_cov_k.device)
    
        # Compute the posterior covariance
        if N_k > 0:
            return (psi + data_cov_k * N_k.to(data_cov_k.device)) / (self.niw_nu + N_k.to(data_cov_k.device) + D  )#+ 2
        else:
            return psi

            
    def compute_post_cov_DPM(self, N_k, mu_k, data_cov_k):
        # If it is hard assignments: N_k is the number of points assigned to cluster K, x_mean is their average
        # If it is soft assignments: N_k is the r_k, the sum of responses to the k-th cluster, x_mean is the data average (all the data)
        D = len(mu_k)
        if N_k > 0:
            return (
                self.niw_psi
                + data_cov_k * N_k  # unnormalize
                + (
                    ((self.niw_kappa * N_k) / (self.niw_kappa + N_k))
                    * ((mu_k - self.niw_m).unsqueeze(1) * (mu_k - self.niw_m).unsqueeze(0))
                )
            ) / (self.niw_nu + N_k + D + 2)
        else:
            return self.niw_psi
    
    

    def Iw_logLikelihoodMarginalized_ADEL(
        self,
        niw_nu,
        niw_psi,
        Nk,
        D,
        sigmak
    ):
        # 1) force CPU
        device = torch.device("cpu")
    
        # 2) ensure all inputs are CPU-tensors
        niw_nu   = torch.tensor(niw_nu,   dtype=torch.float32, device=device)
        Nk       = torch.tensor(Nk,       dtype=torch.float32, device=device)
        D        = int(D)  # dimension can stay Python int
        LOG_PI   = torch.log(torch.tensor(np.pi, device=device))
    
        niw_psi  = torch.tensor(niw_psi,  dtype=torch.float32, device=device)
        if isinstance(sigmak, torch.Tensor):
            sigmak = sigmak.to(device)
        else:
            sigmak = torch.tensor(sigmak, dtype=torch.float32, device=device)
    
        # 3) compute log-determinants
        log_det_psi    = torch.linalg.slogdet(niw_psi)[1]
        log_det_sigmak = torch.linalg.slogdet(sigmak)[1]
    
        # 4) data covariance term
        data_cov_k = (sigmak * (niw_nu + Nk + D) - niw_psi) / Nk
    
        # 5) adjusted determinant
        log_det_adjusted = torch.linalg.slogdet(niw_psi + data_cov_k * Nk)[1]
    
        # 6) assemble log-likelihood
        ll = 0.5 * niw_nu * log_det_psi
        ll -= 0.5 * D * Nk * LOG_PI
        ll -= 0.5 * (niw_nu + Nk) * log_det_adjusted
    
        # 7) multivariate gamma terms
        ll += torch.tensor(multigammaln(0.5 * (niw_nu + Nk), D),
                           dtype=torch.float32, device=device)
        ll -= torch.tensor(multigammaln(0.5 * niw_nu,D),
                           dtype=torch.float32, device=device)
    
        return ll  # already on CPU

    def Iw_logLikelihoodMarginalized(self, niw_nu, niw_psi, Nk, D, sigmak):
        # Ensure niw_psi is a PyTorch tensor
        LOG_PI = torch.log(torch.tensor(np.pi))
        niw_psi = torch.tensor(niw_psi, dtype=torch.float32)
        
        # Compute the log determinant of the scale matrix niw_psi
        log_det_psi = torch.linalg.slogdet(niw_psi)[1]

        # Compute the log determinant of `sigmak`
        log_det_sigmak = torch.linalg.slogdet(sigmak)[1]

        # Use the isolated `data_cov_k` expression to replace the scatter matrix
        data_cov_k = (
            (sigmak * (niw_nu + Nk + D ) - niw_psi) / Nk
        )

        # Compute the log determinant of `niw_psi + data_cov_k * Nk`
        log_det_adjusted = torch.linalg.slogdet(niw_psi + data_cov_k * Nk)[1]

        # Compute the log-likelihood
        log_likelihood = 0.5 * niw_nu * log_det_psi
        log_likelihood -= 0.5 * D * Nk * LOG_PI
        log_likelihood -= 0.5 * (niw_nu + Nk) * log_det_adjusted.item()
        
        # Include terms from multivariate gamma functions
        log_likelihood += torch.tensor(multigammaln(0.5 * (niw_nu + Nk), D), dtype=torch.float32)
        log_likelihood -= torch.tensor(multigammaln(0.5 * niw_nu, D), dtype=torch.float32)
        
        return log_likelihood
    
    """
    def Iw_logLikelihoodMarginalized(self, niw_nu, niw_psi, Nk, D, sigmak):
        
        #NIW-marginalized log-likelihood on CPU, robust to shape/device.
        #- Forces all inputs to CPU
        #- Forces scalar inputs to be 0-dim tensors (shape [])
        #- Avoids in-place ops to prevent broadcast errors
        #- Light SPD guards for slogdet
        #
        device = torch.device('cpu')
    
        # dtype: preserve float64 if provided, else float32
        in_dtype = sigmak.dtype if isinstance(sigmak, torch.Tensor) else torch.float64
        dtype = torch.float64 if in_dtype == torch.float64 else torch.float32
    
        # Tensors on CPU
        sigmak  = torch.as_tensor(sigmak,  dtype=dtype, device=device)
        niw_psi = torch.as_tensor(niw_psi, dtype=dtype, device=device)
    
        # Force true scalars (0-dim) for scalar quantities
        niw_nu  = torch.as_tensor(niw_nu,  dtype=dtype, device=device).reshape([])
        Nk      = torch.as_tensor(Nk,      dtype=dtype, device=device).reshape([])
        D       = torch.as_tensor(D,       dtype=dtype, device=device).reshape([])
    
        LOG_PI = torch.tensor(math.log(math.pi), dtype=dtype, device=device)
    
        # --- SPD symmetrization & tiny regularization guard for slogdet ---
        def _sym(M):
            return 0.5 * (M + M.T)
    
        def _slogdet_spd(M):
            M = _sym(M)
            sign, ld = torch.linalg.slogdet(M)
            if sign <= 0:
                eps = torch.finfo(dtype).eps
                M = M + eps * torch.eye(M.shape[-1], dtype=dtype, device=device)
                sign, ld = torch.linalg.slogdet(M)
            return ld
    
        log_det_psi    = _slogdet_spd(niw_psi)
        # optional: not used but kept for parity/debugging
        # log_det_sigmak = _slogdet_spd(sigmak)
    
        # data covariance surrogate
        # Assumes Nk > 0
        data_cov_k = (sigmak * (niw_nu + Nk + D) - niw_psi) / Nk
    
        adjusted = niw_psi + data_cov_k * Nk
        log_det_adjusted = _slogdet_spd(adjusted)
    
        # --- Assemble without in-place ops ---
        ll = 0.5 * niw_nu * log_det_psi
        ll = ll - 0.5 * D * Nk * LOG_PI
        ll = ll - 0.5 * (niw_nu + Nk) * log_det_adjusted
    
        # SciPy returns Python float; wrap as tensor on CPU with same dtype
        lgm_add = torch.tensor(
            multigammaln(0.5 * (niw_nu.item() + Nk.item()), int(D.item())),
            dtype=dtype, device=device
        )
        lgm_sub = torch.tensor(
            multigammaln(0.5 * niw_nu.item(), int(D.item())),
            dtype=dtype, device=device
        )
        ll = ll + lgm_add - lgm_sub
    
        # Return scalar 0-dim tensor
        return ll.reshape([])"""


    def Iw_logLikelihoodMarginalized_v1(self,niw_nu, niw_psi, Nk, D, sigmak):
        # Ensure niw_psi is a PyTorch tensor
        niw_psi = torch.tensor(niw_psi, dtype=torch.float32)
        
        # Compute the log determinant of the scale matrix niw_psi
        log_det_psi = torch.linalg.slogdet(niw_psi)[1]
        
        # Compute the term involving the determinant of (niw_psi + adjusted sigma_k)
        adjusted_sigma_k = ((sigmak * (niw_nu + Nk + D + 2)) - niw_psi) / Nk
        
        log_det_adjusted = torch.linalg.slogdet(niw_psi + adjusted_sigma_k)[1]
        #adjusted_sigma_k=sigmak
        #log_det_adjusted=  torch.linalg.slogdet(adjusted_sigma_k)[1]
        # Compute the log likelihood
        log_likelihood = 0.5 * niw_nu * log_det_psi
        log_likelihood -= 0.5 * D * Nk * torch.log(torch.tensor(np.pi, dtype=torch.float32))
        #print("log_likelihood",log_likelihood)
        #print("log_det_adjusted" ,log_det_adjusted)
        #print("niw_nu",niw_nu)
        #print("Nk",Nk)
        #print("log_det_adjusted",log_det_adjusted)
        #niw_nu_tensor = torch.tensor([niw_nu], dtype=torch.float64)  # Assuming you want double precision
        #Nk_tensor = torch.tensor([Nk], dtype=torch.float64)

        # Make sure log_det_adjusted is also a scalar tensor of the same dtype
        #log_det_adjusted = torch.tensor([log_det_adjusted], dtype=torch.float64)
        #log_likelihood -= 0.5 * (niw_nu + Nk) * log_det_adjusted
        #log_likelihood -= 0.5 * (niw_nu_tensor + Nk_tensor) * log_det_adjusted
        # For the multigammaln function, we still use scipy as PyTorch does not have a direct equivalent
        log_likelihood += torch.tensor(multigammaln(0.5 * (niw_nu + Nk), D), dtype=torch.float32)
        log_likelihood -= torch.tensor(multigammaln(0.5 * niw_nu, D), dtype=torch.float32)
        log_likelihood -= 0.5 * (niw_nu + Nk) * log_det_adjusted.item()
        return log_likelihood
    
    def logSurfaceArea(self,D):
        LOG_2 = math.log(2)
        LOG_PI = math.log(math.pi)
        return LOG_2 + 0.5 * D * LOG_PI - gammaln(0.5 * D)
    
    def log_marginal_likelihood(self,codes_k,sigmak):
        (N_k, D) = codes_k.shape
        
        return self.Iw_logLikelihoodMarginalized(self.niw_nu, self.niw_psi, N_k, D, sigmak) - self.logSurfaceArea(D)
    
    def log_marginal_likelihood_dynamic_psi(self,codes_k,sigmak,psi_index=None,psi_value=None):
        (N_k, D) = codes_k.shape
        
        if psi_value == None:
          return self.Iw_logLikelihoodMarginalized(self.niw_nu, self.niw_psi_clusters[psi_index], N_k, D, sigmak) - self.logSurfaceArea(D)
        else :
          return self.Iw_logLikelihoodMarginalized(self.niw_nu, psi_value, N_k, D, sigmak) - self.logSurfaceArea(D)
    
    def log_marginal_likelihood_DPM(self, codes_k, mu_k):
        kappa_star, nu_star, mu_0_star, psi_star = self.compute_params_post(
            codes_k, mu_k
        )
        (N_k, D) = codes_k.shape
        return (
            -(N_k * D / 2.0) * np.log(np.pi)
            + mvlgamma(torch.tensor(nu_star / 2.0), D)
            - mvlgamma(torch.tensor(self.niw_nu) / 2.0, D)
            + (self.niw_nu / 2.0) * torch.logdet(self.niw_psi)
            - (nu_star / 2.0) * torch.logdet(psi_star)
            + (D / 2.0) * (np.log(self.niw_kappa) - np.log(kappa_star))
        )
    def log_pdf_multivariate_normal(self,X, Sigma):
        """
        Compute the log PDF of a multivariate normal distribution with mean vector 0.
        """
        k = Sigma.size(0)
        Sigma_inv = torch.linalg.inv(Sigma)
        det_Sigma = torch.linalg.det(Sigma)
        #log_term = torch.log((2 * torch.pi) ** k * det_Sigma)
        log_term = k * torch.log(torch.tensor(2 * torch.pi, dtype=det_Sigma.dtype, device=det_Sigma.device)) + torch.log(det_Sigma + 1e-8)

        X = X.float()
        Sigma_inv = Sigma_inv.float()

        quadratic_form = torch.diagonal(X @ Sigma_inv @ X.T)
        log_pdf = -0.5 * (quadratic_form + log_term)
        
        return log_pdf
    
    def log_mean_projected_data(self, codes, Sigma_k):
        # Forcer les types pour éviter les erreurs float64 vs float32
        print('INTO LOG MEAN PROJ')
        print('codes.size():',codes.size())
        print('simga_k.size() :',Sigma_k.size())
        codes = codes.float()
        Sigma_k = Sigma_k.float()
        self.niw_psi = self.niw_psi.float()
    
        N_k, D = codes.shape
    
        # 1. Calcul du sigma ajusté
        #adjusted_sigma_k = ((Sigma_k * (self.niw_nu + N_k + D)) - self.niw_psi) / N_k
        adjusted_sigma_k=Sigma_k
        adjusted_sigma_k = adjusted_sigma_k.float()  # sécurité
        
        if adjusted_sigma_k.dim() == 3 and adjusted_sigma_k.shape[0] == 1:
           adjusted_sigma_k = adjusted_sigma_k.squeeze(0)
        # 2. Calcul de la moyenne de Karcher
        mean_cluster = KarcherMean(soft_assign=None, codes=codes, cov=None)  # [1, D]
        mean_cluster_vec = mean_cluster.squeeze(0).float()  # [D]
    
        # 3. Échantillonnage m ~ N(0, S?)
        loc = torch.zeros(D, device=adjusted_sigma_k.device, dtype=torch.float32)
        mvn = torch.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=adjusted_sigma_k
        )
        m = mvn.rsample()  # [D]
    
        # 4. Projection de m dans l'espace tangent de la sphère
        #proj = torch.dot(m, mean_cluster_vec) * mean_cluster_vec  # projection sur mean_cluster
        #m_tangent = m - proj  # projection orthogonale
    
        # 5. Évaluation de la log-vraisemblance sous la gaussienne
        log_prob = self.log_pdf_multivariate_normal(m.unsqueeze(0), adjusted_sigma_k).squeeze(0)
    
        return log_prob


        
    def log_mean_projected_data_previous(self,codes, mu_k, Sigma_k):
        # Ensure codes are normalized
        codes_normalized = F.normalize(codes, p=2, dim=-1)
        
        # Project the data onto the tangent space
        projected_data = Log_mapping(codes_normalized, mu_k)
        # Compute the mean of the projected data
        mean_projected_data = torch.mean(projected_data, dim=0, keepdim=True)
        
        # Adjust log_pdf_multivariate_normal to handle mean as a single sample
        log_pdf_value = self.log_pdf_multivariate_normal(mean_projected_data, Sigma_k)
        
        return log_pdf_value

class NIG_prior:
    """A class used to store nig parameters and compute posteriors.
    Used as a class in case we will want to update these parameters.
    The NIG will model each codes channel separetly, so we will have d-dimensions for every hyperparam
    """

    def __init__(self, hparams, codes_dim):
        self.name = "NIG"
        self.dim = codes_dim
        self.prior_mu_0_choice = hparams.prior_mu_0
        self.nig_V = torch.ones(self.dim) / hparams.prior_kappa
        self.nig_a = torch.ones(self.dim) * (hparams.NIW_prior_nu / 2.0)
        self.prior_sigma_choice = hparams.prior_sigma_choice
        if self.prior_sigma_choice == "iso_005":
            self.nig_sigma_sq_0 = torch.ones(self.dim) * 0.005
        if self.prior_sigma_choice == "iso_0001":
            self.nig_sigma_sq_0 = torch.ones(self.dim) * 0.0001

        self.nig_b = torch.ones(self.dim) * (hparams.NIW_prior_nu * self.nig_sigma_sq_0 / 2.0)

    def init_priors(self, codes):
        if self.prior_mu_0_choice == "data_mean":
            self.nig_m = codes.mean(axis=0)
        return self.nig_m, torch.eye(codes.shape[1]) * self.nig_sigma_sq_0

    def compute_params_post(self, codes_k, mu_k=None):
        N = len(codes_k)

        V_star = self.nig_V * (1. / (1 + self.nig_V * N))
        m_star = V_star * (self.nig_m/self.nig_V + codes_k.sum(axis=0))
        a_star = self.nig_a + N / 2.
        b_star = self.nig_b + 0.5 * ((self.nig_m ** 2) / self.nig_V + (codes_k ** 2).sum(axis=0) - (m_star ** 2) / V_star)
        return V_star, m_star, a_star, b_star

    def compute_post_mus(self, N_ks, data_mus):
        # kappa = 1.0 / self.nig_V
        # return ((N_ks.reshape(-1, 1) * data_mus) + (kappa * self.nig_m)) / (
        #     N_ks.reshape(-1, 1) + kappa
        # )

        # for each K we are going to have mu in R^D
        return ((N_ks.reshape(-1, 1) * data_mus) + (1 / self.nig_V * self.nig_m)) / (
            N_ks.reshape(-1, 1) + 1 / self.nig_V
        )

    def compute_post_cov(self, N_ks, mus, data_stds):
        # If it is hard assignments: N_k is the number of points assigned to cluster K, x_mean is their average
        # If it is soft assignments: N_k is the r_k, the sum of responses to the k-th cluster, x_mean is the data average (all the data)

        # N_ks is a d-dimentionl tensor wirh N_ks[d] = N_k of the above
        # data_std is a d-dimentional tensor with data_std[d] is the weighted std along dimention d
        if N_ks > 0:
            post_sigma_sq = (
                data_stds * N_ks
                + 2 * self.nig_b
                + 1 / self.nig_V * ((self.nig_m - mus) ** 2)
            ) / (N_ks + 2 * self.nig_a + 3)
            return post_sigma_sq
        else:
            return torch.eye(mus.shape[1]) * self.nig_sigma_sq_0

    def log_marginal_likelihood(self, codes_k, mu_k):
        # Hard assignment
        # Since we consider the channels to be independent, the log likelihood will be the sum of log likelihood per channel
        V_star, m_star, a_star, b_star = self.compute_params_post(codes_k, mu_k)
        N = len(codes_k)
        lm_ll = 0
        for d in range(self.dim):
            lm_ll += 0.5 * (torch.log(torch.abs(V_star[d])) - torch.log(torch.abs(self.nig_V[d]))) \
                  + self.nig_a[d] * torch.log(self.nig_b[d]) - a_star[d] * torch.log(b_star[d]) \
                  + lgamma(a_star[d]) - lgamma(self.nig_a[d]) - (N / 2.) * torch.log(torch.tensor(np.pi)) - N * torch.log(torch.tensor(2.))
        return lm_ll
