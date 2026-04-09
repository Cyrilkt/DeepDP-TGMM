#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import umap
import torch
from torch import optim
import pytorch_lightning as pl
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, silhouette_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure

from src.clustering_models.clusternet_modules.utils.plotting_utils import PlotUtils
from src.clustering_models.clusternet_modules.utils.training_utils import training_utils
from src.clustering_models.clusternet_modules.utils.clustering_utils.priors import (
    Priors,
)
from src.clustering_models.clusternet_modules.utils.clustering_utils.clustering_operations import (
    init_mus_and_covs,
    compute_data_covs_hard_assignment,
    ensure_positive_definite,
    KarcherMean,
)
from src.clustering_models.clusternet_modules.utils.clustering_utils.split_merge_operations import (
    update_models_parameters_split,
    split_step,
    merge_step,
    update_models_parameters_merge,
)
from src.clustering_models.clusternet_modules.models.Classifiers import MLP_Classifier, Subclustering_net

##Added by me 
import torch.nn.functional as F
from datetime import datetime
import torch.nn as nn
import string
from torch.distributions.constraints import positive_definite


class ClusterNetModel(pl.LightningModule):
    def __init__(self, hparams, input_dim, init_k, feature_extractor=None, n_sub=2, centers=None, init_num=0):
        """The main class of the unsupervised clustering scheme.
        Performs all the training steps.

        Args:
            hparams ([namespace]): model-specific hyperparameters
            input_dim (int): the shape of the input data
            train_dl (DataLoader): The dataloader to train on
            init_k (int): The initial K to start the net with
            feature_extractor (nn.Module): The feature extractor to get codes with
            n_sub (int, optional): Number of subclusters per cluster. Defaults to 2.

        """

        super().__init__()
        self.hparams = hparams
        #print("HPARAMS :",hparams)
        self.K = init_k
        # new attribute, will hold predicted K at epoch end
        self.K_pred = None
        self.n_sub = hparams.n_sub
        self.codes_dim = input_dim
        self.split_performed_2=False #To remove
        self.split_performed = False  # indicator to know whether a split was performed
        self.merge_performed = False
        self.feature_extractor = feature_extractor
        #print('REINIT WITH MUS :',hparams.reinit_with_mus)
        #print('prior_mu_0 : ',hparams.prior_mu_0)
        self.centers = centers #if hparams.reinit_with_mus  else None
        print('SELF CENTERS :',self.centers)
        if self.hparams.seed:
            pl.utilities.seed.seed_everything(self.hparams.seed)

        # initialize cluster_net
        self.cluster_net = MLP_Classifier(hparams, k=self.K, codes_dim=self.codes_dim)
        
        #Now need to be create in init_mus_covs:
        #if not self.hparams.ignore_subclusters:
            # initialize subclustering net
        #    self.subclustering_net = Subclustering_net(hparams, codes_dim=self.codes_dim, k=self.K)
        #else:
        #    self.subclustering_net = None
        self.subclustering_net = None
        self.subcluster_is_initialized=False
        self.last_key = self.K - 1  # variable to help with indexing the dict
        self.debug=hparams.debug
        self.training_utils = training_utils(hparams)
        self.last_val_NMI = 0
        self.init_num = init_num
        self.prior_sigma_scale = self.hparams.prior_sigma_scale
        if self.init_num > 0 and self.hparams.prior_sigma_scale_step != 0:
            self.prior_sigma_scale = self.hparams.prior_sigma_scale / (self.init_num * self.hparams.prior_sigma_scale_step)
        self.use_priors = self.hparams.use_priors
        self.prior_sigma_choice=self.hparams.prior_sigma_choice
        self.prior = Priors(hparams, K=self.K, codes_dim=self.codes_dim, prior_sigma_scale=self.prior_sigma_scale,prior_choice=self.prior_sigma_choice) # we will use for split and merges even if use_priors is false
        self.mus_inds_to_merge = None
        self.mus_ind_to_split = None
        # Initialize label mappings
        uppercase_letters = list(string.ascii_uppercase)
        num_letters = len(uppercase_letters)
        self.cluster_labels = {}
        self.prior_labels = {}
        self.subcluster_labels = {}  # Will be initialized later when n_sub_list is available
        #self.merge_not_performed=True
        self.origin_cardinality=None
        for i in range(self.K):
            quotient, remainder = divmod(i, num_letters)
            label = uppercase_letters[remainder] * (quotient + 1)
            self.cluster_labels[i] = label
            self.prior_labels[i] = label

        self.n_sub_list = []  # Will be initialized later
        self.run_best_acc = float('-inf')
        self.run_best_nmi = float('-inf')
        self.run_best_ari = float('-inf')
        self.best_epoch = None
        print('USE PRIORS IS :', self.hparams.use_priors)


    def forward(self, x,use_feature_extractor=True):
        print(x)
        if self.feature_extractor is not None and use_feature_extractor:
            if self.args.dataset=='fashionmnist':
              with torch.no_grad():
                  codes = torch.from_numpy(
                      self.feature_extractor(x, latent=True)
                  ).to(device=self.device)
            else:
              with torch.no_grad():
                  codes = torch.from_numpy(
                      self.feature_extractor(x.view(x.size()[0], -1), latent=True,infer=True)
                  ).to(device=self.device)
        else:
            codes = x
        codes=F.normalize(codes,p=2,dim=-1).to(device=self.device)
        #print("self.cluster_net(codes): ",self.cluster_net(codes).size())
        return self.cluster_net(codes)

        
    
    def on_train_epoch_start(self):
        # get current training_stage
        self.current_training_stage = (
            "gather_codes" if self.current_epoch == 0 and not hasattr(self, "mus") else "train_cluster_net"
        )
        if not self.hparams.ignore_subclusters and self.subclustering_net is not None and not self.subcluster_is_initialized:
            print('INTO REASSIGN SUBCLUSTERING OPTIMIZER')
            sub_clus_opt = optim.Adam(self.subclustering_net.parameters(), lr=self.hparams.subcluster_lr)
            sub_clus_opt = self.trainer.optimizers[self.optimizers_dict_idx["subcluster_net_opt"]]
            sub_clus_opt.param_groups = []  # Clear existing dummy parameter group
            sub_clus_opt.add_param_group({"params": self.subclustering_net.parameters()})
            self.subcluster_is_initialized=True
            
        self.initialize_net_params(stage="train")
        if self.split_performed or self.merge_performed:
            self.split_performed = False
            self.merge_performed = False

    def on_validation_epoch_start(self):
        self.initialize_net_params(stage="val")
        return super().on_validation_epoch_start()

    def initialize_net_params(self, stage="train"):
        self.codes = []
        if stage == "train":
            if self.current_epoch > 0:
                del self.train_resp, self.train_resp_sub, self.train_gt
            self.train_resp = []
            self.train_resp_sub = []
            self.train_gt = []
        else:
            if self.current_epoch > 0:
                del self.val_resp, self.val_resp_sub, self.val_gt
            self.val_resp = []
            self.val_resp_sub = []
            self.val_gt = []
    """
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        #print("TRAINING8STEP X, Y :", x.size(),y.size())
        #print(batch.meta_data)
        if self.feature_extractor is not None:
            with torch.no_grad():
                codes = torch.from_numpy(
                    self.feature_extractor(x.view(x.size()[0], -1), latent=True)
                ).to(device=self.device)
                codes=F.normalize(codes,p=2,dim=-1).to(device=self.device)
                
        else:
            codes = x
            codes=F.normalize(codes,p=2,dim=-1).to(device=self.device)
            
        if self.current_training_stage == "gather_codes":
            return self.only_gather_codes(codes, y, optimizer_idx)

        elif self.current_training_stage == "train_cluster_net":
            if not self.hparams.ignore_subclusters and self.subclustering_net is not None and not self.subcluster_is_initialized:
                sub_clus_opt = optim.Adam(self.subclustering_net.parameters(), lr=self.hparams.subcluster_lr)
                self.optimizers_dict_idx["subcluster_net_opt"] = 1
                self.trainer.optimizers.append(sub_clus_opt)
                self.subcluster_is_initialized=True
            #print("into training step train cluster_net")
            return self.cluster_net_pretraining(codes, y, optimizer_idx, x if batch_idx == 0 else None)

        else:
            raise NotImplementedError()"""
    def log_clustering_params(self):
        print(f"Epoch {self.current_epoch} - Cluster Labels:")
        for idx, label in self.cluster_labels.items():
            print(f"Cluster MLP index {idx}: Label '{label}'")

        if not self.hparams.ignore_subclusters and self.subcluster_labels:
            print(f"Epoch {self.current_epoch} - Subcluster Labels:")
            for cluster_idx, sub_labels in self.subcluster_labels.items():
                cluster_label = self.cluster_labels.get(cluster_idx, f"Cluster {cluster_idx}")
                print(f"Cluster '{cluster_label}' (MLP index {cluster_idx}):")
                for sub_idx, sub_label in sub_labels.items():
                    print(f"  Subcluster index {sub_idx}: Label '{sub_label}'")

        print(f"Epoch {self.current_epoch} - Prior Labels:")
        for idx, label in self.prior_labels.items():
            print(f"Prior {idx}: Label '{label}'")
          
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        #with torch.autograd.detect_anomaly():
        x, y = batch
        #print("TRAINING8STEP X, Y :", x.size(),y.size())
        #print(batch.meta_data)
        if self.feature_extractor is not None:
            with torch.no_grad():
                codes = torch.from_numpy(
                    self.feature_extractor(x.view(x.size()[0], -1), latent=True)
                ).to(device=self.device)
                codes=F.normalize(codes,p=2,dim=-1).to(device=self.device)
                
        else:
            codes = x
            codes=F.normalize(codes,p=2,dim=-1).to(device=self.device)

        if self.current_training_stage == "gather_codes":
            return self.only_gather_codes(codes, y, optimizer_idx)

        elif self.current_training_stage == "train_cluster_net":
            #print("into training step train cluster_net")
            return self.cluster_net_pretraining(codes, y, optimizer_idx, x if batch_idx == 0 else None,batch_idx=batch_idx)

        else:
            raise NotImplementedError()

    def only_gather_codes(self, codes, y, optimizer_idx):
        """Only log codes for initialization

        Args:
            codes ([type]): The input data in the latent space
            y ([type]): The ground truth labels
            optimizer_idx ([type]): The optimizer index
        """
        # log only once
        if optimizer_idx == self.optimizers_dict_idx["cluster_net_opt"]:
            #print("into only gather")
            (
                self.codes,
                self.train_gt,
                _,
                _,
            ) = self.training_utils.log_codes_and_responses(
                model_codes=self.codes,
                model_gt=self.train_gt,
                model_resp=self.train_resp,
                model_resp_sub=self.train_resp_sub,
                codes=codes,
                y=y,
                logits=None,
            )
        return None
    
    def cluster_net_pretraining_v1(self, codes, y, optimizer_idx, x_for_vis=None):
        codes = codes.view(-1, self.codes_dim)
        logits = self.cluster_net(codes)
        cluster_loss = self.training_utils.cluster_loss_function(
            codes,
            logits,
            model_mus=self.mus,
            K=self.K,
            codes_dim=self.codes_dim,
            model_covs=self.covs if self.hparams.cluster_loss in ("diag_NIG", "KL_GMM_2") else None,
            pi=self.pi,
            logger=self.logger
        )
        
        # Log cluster loss
        self.log(
            "cluster_net_train/train/cluster_loss",
            self.hparams.cluster_loss_weight * cluster_loss,
            on_step=True,
            on_epoch=False,
        )
        
        loss = self.hparams.cluster_loss_weight * cluster_loss
    
        # Subclustering loss if subclustering_net is initialized
        if self.subclustering_net is not None and not self.hparams.ignore_subclusters:
            if optimizer_idx == self.optimizers_dict_idx.get("subcluster_net_opt"):
                # Optimize the subclusters' nets
                logits = logits.detach()
                sublogits = self.subcluster(
    codes, logits, self.n_sub_list, hard_assignment=True, batch_idx=batch_idx, total_batches=len(self.train_dataloader()))

                subcluster_loss = self.training_utils.subcluster_loss_function_new(
                codes,
                logits,
                sublogits,
                self.K,
                self.n_sub_list,  # Pass the list of subcluster counts per cluster
                self.mus_sub,
                covs_sub=self.covs_sub if self.hparams.subcluster_loss in ("diag_NIG", "KL_GMM_2") else None,
                pis_sub=self.pi_sub,
                cluster_labels=self.cluster_labels,
                subcluster_labels=self.subcluster_labels,
                batch_idx=batch_idx,
                total_batches=len(self.train_dataloader()))           
                # Log subcluster loss
                self.log(
                    "cluster_net_train/train/subcluster_loss",
                    self.hparams.subcluster_loss_weight * subcluster_loss,
                    on_step=True,
                    on_epoch=True,
                )
                
                loss = self.hparams.subcluster_loss_weight * subcluster_loss
            else:
                sublogits = None
        else:
            sublogits = None
        
        # Log data if it's the last optimizer
        if optimizer_idx == len(self.optimizers_dict_idx) - 1:
            (
                self.codes,
                self.train_gt,
                self.train_resp,
                self.train_resp_sub,
            ) = self.training_utils.log_codes_and_responses(
                self.codes,
                self.train_gt,
                self.train_resp,
                self.train_resp_sub,
                codes,
                logits.detach(),
                y,
                sublogits=sublogits,
            )
    
        # Return the loss if it was computed
        if loss is not None:
            return loss
        else:
            return None


    def cluster_net_pretraining(
        self,
        codes,
        y,
        optimizer_idx,
        x_for_vis=None,
        batch_idx=None
    ):
        """Pretraining function for the clustering and subclustering nets.
        At this stage, the only loss is the cluster and subcluster loss. The autoencoder weights are held constant.

        Args:
            codes ([type]): The encoded data samples
            y: The ground truth labels
            optimizer_idx ([type]): The pytorch optimizer index
        """
        codes = codes.view(-1, self.codes_dim)
        #codes= F.normalize(codes+torch.randn_like(codes)*0.1,dim=1)
        logits = self.cluster_net(codes)
        
        if batch_idx%50 == 0:
          unique_assignementtocluster=torch.unique(logits.argmax(dim=1))
          #print(f"Batch {batch_idx}: Unique Clusters  assigned: {unique_assignementtocluster.tolist()}")   
        #print('MLP CLASSFIER LOGITS :',logits.size())
        warmup=False
        sub_index=False
        #if self.current_epoch>=10:
        #  warmup=True
        cluster_loss = self.training_utils.cluster_loss_function(
            codes,
            logits,
            model_mus=self.mus,
            K=self.K,
            codes_dim=self.codes_dim,
            model_covs=self.covs if self.hparams.cluster_loss in ("diag_NIG", "KL_GMM_2") else None,
            pi=self.pi,
            logger=self.logger,
            warmup=warmup,
        )
        self.log(
            "cluster_net_train/train/cluster_loss",
            self.hparams.cluster_loss_weight * cluster_loss,
            on_step=True,
            on_epoch=False,
        )
        loss = self.hparams.cluster_loss_weight * cluster_loss
        #if optimizer_idx == self.optimizers_dict_idx["subcluster_net_opt"]:
        #if not self.hparams.ignore_subclusters and optimizer_idx == self.optimizers_dict_idx["subcluster_net_opt"]:
        if optimizer_idx == self.optimizers_dict_idx["subcluster_net_opt"]:             
            # optimize the subclusters' nets
            logits = logits.detach()
            if self.hparams.start_sub_clustering <= self.current_epoch and not self.hparams.ignore_subclusters:
                sublogits, masked_sublogits = self.subcluster(
                    codes,
                    logits,
                    self.n_sub_list,
                    hard_assignment=True,
                    batch_idx=batch_idx,
                    total_batches=len(self.train_dataloader()),
                    debug=True  # Enable debugging
                )
                # Debug: Print subcluster assignments
                #if batch_idx %10 == 0:
                """  
                if batch_idx %50 ==0:
                  subcluster_assignments = sublogits.argmax(dim=1)
                  unique_subclusters = torch.unique(subcluster_assignments)
                  print(f"Batch {batch_idx}: Unique subclusters assigned: {unique_subclusters.tolist()}")"""
                if batch_idx == 0 or batch_idx == len(self.train_dataloader()) - 1:
                  def check_nan(tensor, name):
                      if torch.isnan(tensor).any():
                          print(f"[Batch {batch_idx}] ? NaN detected in {name}")
                      if torch.isinf(tensor).any():
                          print(f"[Batch {batch_idx}] ? Inf detected in {name}")
          
                  check_nan(codes, "codes")
                  check_nan(logits, "logits")
                  check_nan(sublogits, "sublogits")
                  check_nan(self.mus_sub, "mus_sub")
                  check_nan(self.covs_sub, "covs_sub")
                  check_nan(self.pi_sub, "pis_sub")
                #is_split=self.split_performed_2
                subcluster_loss = self.training_utils.subcluster_loss_function_new(
                    codes,
                    logits,
                    sublogits,
                    K=self.K,
                    n_sub_list=self.n_sub_list,
                    mus_sub=self.mus_sub,
                    covs_sub=self.covs_sub,
                    pis_sub=self.pi_sub,
                    cluster_labels=self.cluster_labels,
                    subcluster_labels=self.subcluster_labels,
                    batch_idx=batch_idx,
                    total_batches=len(self.train_dataloader()),
                    masked_subresp=masked_sublogits,  # Pass the masked responses
                )
                """
                subcluster_loss = self.training_utils.subcluster_loss_function_new(
                    codes=codes,
                    logits=logits,
                    subresp=sublogits,                  # <-- match method arg name
                    K=self.K,
                    n_sub_list=self.n_sub_list,
                    mus_sub=self.mus_sub,
                    covs_sub=self.covs_sub,
                    pis_sub=self.pi_sub,
                    covariance=self.hparams.covariance,                # will be ignored by your current method (it hardcodes 'diag')
                )"""
                #is_split=self.split_performed_2  
                self.log(
                    "cluster_net_train/train/subcluster_loss",
                    self.hparams.subcluster_loss_weight * subcluster_loss,
                    on_step=True,
                    on_epoch=True,
                )
                loss = self.hparams.subcluster_loss_weight * subcluster_loss
                if batch_idx == 0 or batch_idx == len(self.train_dataloader()) - 1:
                   print(f"[Batch {batch_idx}] -> subcluster_loss = {subcluster_loss.item():.6f}")
                if torch.isnan(loss).any():
                   print('Nan value detected for subcluster_loss at batch : {batch_idx}')
            else:
                sublogits = None
                #loss=None
                loss =  0.0 * self.dummy_param.sum() #original
                #loss = torch.zeros(1, device=codes.device, requires_grad=True) + 0.0 * self.dummy_param.sum()
                
                sub_index=True
        else:
            sublogits = None
        # log data only once
        if optimizer_idx == len(self.optimizers_dict_idx) - 1:
            (
                self.codes,
                self.train_gt,
                self.train_resp,
                self.train_resp_sub,
            ) = self.training_utils.log_codes_and_responses(
                self.codes,
                self.train_gt,
                self.train_resp,
                self.train_resp_sub,
                codes,
                logits.detach(),
                y,
                sublogits=sublogits,
            )

        
        #if loss is not None:
        #    return loss
        #else:
        #    return None
        #print('into loss return')
        if loss is not None :
            return loss
        else:
           return None
    
    
        


    def validation_step(self, batch, batch_idx):
        x, y = batch
        #print('into validation stepp')

        #print("val X , Y ",x.size(),y.size())
        if self.feature_extractor is not None:
            with torch.no_grad():
                codes = torch.from_numpy(
                    self.feature_extractor(x.view(x.size()[0], -1), latent=True)
                ).to(device=self.device)
                codes=F.normalize(codes,p=2,dim=-1).to(device=self.device)
        else:
            codes = x
            codes=F.normalize(codes,p=2,dim=-1).to(device=self.device)

        logits = self.cluster_net(codes)
        if batch_idx == 0 and (self.current_epoch < 5 or self.current_epoch % 50 == 0):
            self.log_logits(logits)

        if self.current_training_stage != "gather_codes":
            cluster_loss = self.training_utils.cluster_loss_function(
                codes.view(-1, self.codes_dim),
                logits,
                model_mus=self.mus,
                K=self.K,
                codes_dim=self.codes_dim,
                model_covs=self.covs
                if self.hparams.cluster_loss in ("diag_NIG", "KL_GMM_2")
                else None,
                pi=self.pi,
            )
            loss = self.hparams.cluster_loss_weight * cluster_loss
            #print("INTO  LOG CLUSTERLOSS")
            self.log("cluster_net_train/val/cluster_loss", loss)

            if self.current_epoch >= self.hparams.start_sub_clustering and not self.hparams.ignore_subclusters:
                subclusters = self.subcluster(
    codes, logits, self.n_sub_list, hard_assignment=True, batch_idx=batch_idx, total_batches=len(self.train_dataloader())
)

                subcluster_loss = self.training_utils.subcluster_loss_function_new(
                codes,
                logits,
                subclusters,
                self.K,
                self.n_sub_list,
                self.mus_sub,
                covs_sub=self.covs_sub,
                pis_sub=self.pi_sub,
                cluster_labels=self.cluster_labels,
                subcluster_labels=self.subcluster_labels,
                batch_idx=batch_idx,
                total_batches=len(self.train_dataloader()))
                self.log("cluster_net_train/val/subcluster_loss", subcluster_loss)
                
                #subcluster_loss = self.training_utils.subcluster_loss_function_new(
                #    codes=codes,
                #    logits=logits,
                #    subresp=subclusters,                  # <-- match method arg name
                #    K=self.K,
                #    n_sub_list=self.n_sub_list,
                #    mus_sub=self.mus_sub,
                #    covs_sub=self.covs_sub,
                #    pis_sub=self.pi_sub,
                #    covariance=self.hparams.covariance,                # will be ignored by your current method (it hardcodes 'diag')
                #)
                loss += self.hparams.subcluster_loss_weight * subcluster_loss
            else:
                subclusters = None
                #self.merge_not_performed=True
        else:
            loss = torch.tensor(1.0)
            subclusters = None
            logits = None

        # log val data
        (
            self.codes,
            self.val_gt,
            self.val_resp,
            self.val_resp_sub,
        ) = self.training_utils.log_codes_and_responses(
            self.codes,
            self.val_gt,
            self.val_resp,
            model_resp_sub=self.val_resp_sub,
            codes=codes,
            logits=logits,
            y=y,
            sublogits=subclusters,
            stage="val",
        )
        #print("into val_step self.codes ",self.codes.size())
        
        return {"loss": loss}

    
    def on_train_end(self) -> None:
        # this runs exactly once, when training (fit) finishes
        with torch.no_grad():
            # 1) gather all softmax responses from final training epoch
            resp_list   = self.train_resp if isinstance(self.train_resp, (list, tuple)) else [self.train_resp]
            resp_tensor = torch.cat(resp_list, dim=0)    # [N, K_old]
            pred_labels = resp_tensor.argmax(dim=1)      # [N]

            # 2) compute K_pred
            unique_preds  = torch.unique(pred_labels)
            self.K_pred   = int(unique_preds.numel())
            print(f"Final K_pred: {self.K_pred}")

            # 3) mask out any unused clusters
            counts     = torch.bincount(pred_labels, minlength=self.mus.size(0))
            keep_mask  = counts > 0                    # [K_old]

            # 4) prune all per-cluster tensors
            self.mus   = self.mus[keep_mask]           # [K_new, D]
            self.covs  = self.covs[keep_mask]          # [K_new, D, D]
            self.pi    = self.pi[keep_mask]            # [K_new]

            # 5) sync K
            self.K     = self.K_pred
            
    def training_epoch_end(self, outputs):
        """Perform logging operations and computes the clusters' and the subclusters' centers.
        Also perform split and merges steps

        Args:
            outputs ([type]): [description]
        """
        
        #to del 
        #if self.current_epoch==46:
        #    self.weight._backward_hooks.clear()
        if self.current_epoch == self.hparams.train_cluster_net-1:
           #print('INTO PLOT')
           current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
           #PlotUtils.debugging_visualize_embeddings(self.codes.size()[1],self.codes, vae_labels=None if not self.hparams.use_labels_for_eval else self.train_gt,current_epoch=self.current_epoch, UMAP=True, centers=self.mus,fname=f'/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/from_clusternet/embedding_{current_time}.png')
           
        if self.current_training_stage == "gather_codes":
            # Initalize plotting utils
            self.plot_utils = PlotUtils(
                self.hparams, self.logger, self.codes.view(-1, self.codes_dim)
            )
            #A SUPPRIMER
            #torch.save(self.codes,'/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/codes.pt')
            #torch.save(self.train_gt,'/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/gt.pt')
            # first time to compute mus
            #A decommenter
            if self.hparams.prior_sigma_choice != 'dynamic_data_std':
              #self.prior.init_priors(self.codes.view(-1, self.codes_dim),self.train_gt)
              self.prior.init_priors(self.codes.view(-1, self.codes_dim))
            if self.centers is not None:
                print('self.centers :',self.centers)
                # we have initialization from somewhere
                self.mus = torch.from_numpy(self.centers).cpu()
                self.centers = None
                print('BEFORE INIT COVS PIS GIVEN MUS4')
                self.init_covs_and_pis_given_mus()
                self.freeze_mus_after_init_until = self.current_epoch + self.hparams.freeze_mus_after_init
                #self.prior.init_priors(self.codes.view(-1, self.codes_dim),self.train_gt)

            else:
                #Toujours le cas si on est pas en training Deepdpm et qu'on reprend ppas les centres a chaque etape
                self.freeze_mus_after_init_until = 0
                #print("self.codes.size():", self.codes.size())
                
                #self.mus, self.covs, self.pi, init_labels = init_mus_and_covs(
                #    codes=self.codes.view(-1, self.codes_dim),
                #    K=self.K,
                #    how_to_init_mu=self.hparams.how_to_init_mu,
                #    logits=self.train_resp,
                #    use_priors=self.hparams.use_priors,
                #    prior=self.prior,
                #    random_state=0,
                #    device=self.device,
                #)
                self.mus, self.covs, self.pi, init_labels ,self.prior = init_mus_and_covs(
                    codes=self.codes.view(-1, self.codes_dim),
                    K=self.K,
                    how_to_init_mu=self.hparams.how_to_init_mu,
                    logits=self.train_resp,
                    use_priors=self.hparams.use_priors,
                    prior=self.prior,
                    random_state=0,
                    device=self.device,
                    prior_choice=self.hparams.prior_sigma_choice
                )
                if self.hparams.use_labels_for_eval:
                    if (self.train_gt < 0).any():
                        # some samples don't have label, e.g., stl10
                        gt = self.train_gt[self.train_gt > -1]
                        init_labels = init_labels[self.train_gt > -1]
                    else:
                        gt = self.train_gt
                    if len(gt) > 2 * (10 ** 5):
                        # sample only a portion of the codes
                        gt = gt[:2 * (10**5)]
                    print("GT AND init_labels ",gt.size(),init_labels.size())
                    init_nmi = normalized_mutual_info_score(gt, init_labels)
                    init_ari = adjusted_rand_score(gt, init_labels)
                    self.log("cluster_net_train/init_nmi", init_nmi)
                    self.log("cluster_net_train/init_ari", init_ari)
                if self.hparams.log_emb == "every_n_epochs" and (self.current_epoch % self.hparams.log_emb_every == 0 or self.current_epoch == 1):
                    self.plot_utils.visualize_embeddings(
                        self.hparams,
                        self.logger,
                        self.codes_dim,
                        vae_means=self.codes,
                        vae_labels=init_labels,
                        val_resp=None,
                        current_epoch=self.current_epoch,
                        y_hat=None,
                        centers=self.mus,
                        stage="init_Kmeans"
                    )
                if self.current_epoch == 0 and (self.hparams.log_emb in ("every_n_epochs", "only_sampled") and self.current_epoch % self.hparams.log_emb_every == 0):
                    perm = torch.randperm(self.train_gt.size(0))
                    idx = perm[:10000]
                    sampled_points = self.codes[idx]
                    sampled_labeled = self.train_gt[idx] if self.hparams.use_labels_for_eval else None
                    self.plot_utils.visualize_embeddings(
                        self.hparams,
                        self.logger,
                        self.codes_dim,
                        vae_means=sampled_points,
                        vae_labels=sampled_labeled,
                        val_resp=None,
                        current_epoch=self.current_epoch,
                        y_hat=None,
                        centers=None,
                        training_stage='train_sampled',
                        UMAP=False
                    )

        else:
            # add avg loss of all losses
            if not self.hparams.ignore_subclusters:
                clus_losses, subclus_losses = outputs[0], outputs[1]
            else:
                #clus_losses = outputs[0], outputs[1] 
                clus_losses = outputs[0]  #original
            avg_clus_loss = torch.stack([x["loss"] for x in clus_losses]).mean()
            self.log("cluster_net_train/train/avg_cluster_loss", avg_clus_loss)
            if self.current_epoch >= self.hparams.start_sub_clustering and not self.hparams.ignore_subclusters:
                avg_subclus_loss = torch.stack([x["loss"] for x in subclus_losses]).mean()
                self.log("cluster_net_train/train/avg_subcluster_loss", avg_subclus_loss)

            # Compute mus and perform splits/merges
            #print('COMPUTE')
            """ original
            perform_split = self.training_utils.should_perform_split(
                self.current_epoch
            ) and self.centers is None 
            print("assessing if we should enter hasting ratio split assessment:",perform_split)
            perform_merge = self.training_utils.should_perform_merge(
                self.current_epoch,
                self.split_performed
            ) and self.centers is None
            """
            """ORIGNAL V2
            perform_split = self.training_utils.should_perform_split(
                self.current_epoch
            ) and self.centers is None and not self.hparams.ignore_subclusters
            print("assessing if we should enter hasting ratio split assessment:",perform_split)
            perform_merge = self.training_utils.should_perform_merge(
                self.current_epoch,
                self.split_performed
            ) and self.centers is None and not self.hparams.ignore_subclusters
            
            """
            mode = getattr(self.hparams, "split_merge_mode", "both")
            # Conditions “brutes” comme avant
            split_due = self.training_utils.should_perform_split(self.current_epoch)
            merge_due = self.training_utils.should_perform_merge(
                self.current_epoch,
                self.split_performed,
            )
            
            # Condition commune (pas de centers, pas d’ignore_subclusters)
            base_ok = self.centers is None and not self.hparams.ignore_subclusters
            
            if mode == "both":
                # Comportement original : split et merge alternent
                perform_split = split_due and base_ok
                perform_merge = merge_due and base_ok
            
            elif mode == "split_only":
                # On remplace les phases de merge par des phases de split :
                # si une phase split OU merge est due, on fait un split.
                perform_split = (split_due or merge_due) and base_ok
                perform_merge = False
            
            elif mode == "merge_only":
                # On remplace les phases de split par des phases de merge :
                # si une phase split OU merge est due, on fait un merge.
                perform_split = False
                perform_merge = (merge_due or split_due) and base_ok
            
            else:
                raise ValueError(f"Unknown split_merge_mode: {mode}")
            
            print("assessing if we should enter hasting ratio split assessment:", perform_split)
            print("assessing if we should enter hasting ratio merge assessment:", perform_merge)
            
            #A supprimer
            #perform_merge= False
            print("assessing if we should enter hasting ratio merge assessment:",perform_merge)
            # do not compute the mus in the epoch(s) following a split or a merge
            if self.centers is not None:
                # we have initialization from somewhere
                self.mus = torch.from_numpy(self.centers).cpu()
                self.centers = None
                self.init_covs_and_pis_given_mus()
                self.freeze_mus_after_init_until = self.current_epoch + self.hparams.freeze_mus_after_init
            freeze_mus = self.training_utils.freeze_mus(
                self.current_epoch,
                self.split_performed
            ) or self.current_epoch <= self.freeze_mus_after_init_until
            #print("self.training_utils.freeze_mus: ",self.training_utils.freeze_mus(
            #     self.current_epoch,
            #     self.split_performed
            #))
            #print("self.current_epoch <= self.freeze_mus_after_init_until :",self.current_epoch <= self.freeze_mus_after_init_until)
            print("Assessing if freeze_mus or curr epoch inferior to nb of epoch to freeze mus after ini: ",freeze_mus)
            #if not freeze_mus:

            if not freeze_mus :
                """
                (
                    self.pi,
                    self.mus,
                    self.covs,
                ) = self.training_utils.comp_cluster_params(
                    self.train_resp,
                    self.codes.view(-1, self.codes_dim),
                    self.pi,
                    self.K,
                    self.prior,
                )"""
                (
                    self.pi,
                    self.mus,
                    self.covs,
                ) = self.training_utils.comp_cluster_params(
                    self.train_resp,
                    self.codes.view(-1, self.codes_dim),
                    self.pi,
                    self.K,
                    self.covs,
                    self.prior,
                    
                )
                #print('outside comp cluster mus :',self.mus.size())

            if (self.hparams.start_sub_clustering == self.current_epoch + 1) or (self.hparams.ignore_subclusters and (perform_split or perform_merge)):
                if not self.hparams.ignore_subclusters: 
                    (
                        self.pi_sub,
                        self.mus_sub,
                        self.covs_sub,
                        self.subclustering_net,
                        self.n_sub_list
                    ) = self.training_utils.init_subcluster_params(
                        train_resp=self.train_resp,
                        train_resp_sub=self.train_resp_sub,
                        codes=self.codes.view(-1, self.codes_dim),
                        K=self.K,
                        n_sub=self.n_sub,
                        prior=self.prior,
                        mus=self.mus                   
                    ) # I added self.mus 
                    self.subclustering_net=self.subclustering_net.to(device=self.device)
                    self.subcluster_labels = {}
                    for i in range(self.K):
                        label = self.cluster_labels[i]
                        self.subcluster_labels[i] = {j: label + str(j + 1) for j in range(self.n_sub_list[i])}
            
            elif (
                self.hparams.start_sub_clustering <= self.current_epoch
                and not freeze_mus and not self.hparams.ignore_subclusters and not self.hparams.nofsub
            ): 
                (
                    self.pi_sub,
                    self.mus_sub,
                    self.covs_sub,
                ) = self.training_utils.comp_subcluster_params(
                    self.train_resp,
                    self.train_resp_sub,
                    self.codes,
                    self.mus,
                    self.K,
                    self.n_sub_list,
                    self.mus_sub,
                    self.covs_sub,
                    self.pi_sub,
                    self.prior,
                )
            
            print('FREEZE_MUS',freeze_mus)
            print('perform_split',perform_split)
             
            if perform_split and not freeze_mus :
                # perform splits
                self.training_utils.last_performed = "split"
                print("###INTO SPLIT  STEP ")
                split_decisions, accepted_subclusters = split_step(
                    self.K,
                    self.codes,
                    self.train_resp,
                    self.train_resp_sub,
                    self.covs,
                    self.covs_sub,
                    self.hparams.cov_const,
                    self.n_sub_list,
                    self.hparams.alpha,
                    self.hparams.split_prob,
                    self.prior,
                    self.hparams.ignore_subclusters,
                    mus=self.mus,
                    mus_sub=self.mus_sub
                )
                if split_decisions.any() and not self.hparams.nosplit: # ajout de moi : and not self.hparams.nosplit
                    self.split_performed = True
                    self.split_performed_2=True
                    self.split_epoch_occur=self.current_epoch
                    print("split decisions has been made for at least one of the cluster")
                    #self.perform_split_operations(split_decisions)
                    self.perform_split_operations(split_decisions, accepted_subclusters)
            
            print('BEFORE perform_merge and not freezemus cdt' , perform_merge , ' and', not freeze_mus)
            if perform_merge and not freeze_mus :
                # make sure no split and merge step occur in the same epoch
                # perform merges
                # =1 to be one epoch after a split
                #print("merged decisions has been made ")
                print('INTO MERGE STEP')
                self.training_utils.last_performed = "merge"
                """
                mus_to_merge, highest_ll_mus = merge_step(
                    self.mus,
                    self.train_resp,
                    self.codes,
                    self.K,
                    self.hparams.raise_merge_proposals,
                    self.hparams.cov_const,
                    self.hparams.alpha,
                    self.hparams.merge_prob,
                    prior=self.prior,
                    covs=self.covs
                )"""
                print('SELF.N_SUB_LIST: ',self.n_sub_list)
                mus_to_merge, highest_ll_mus, clusters_to_suppress = merge_step(
                  mus=self.mus,
                  logits=self.train_resp,
                  codes=self.codes,
                  K=self.K,
                  raise_merge_proposals=self.hparams.raise_merge_proposals,
                  cov_const=self.hparams.cov_const,
                  alpha=self.hparams.alpha,
                  merge_prob=self.hparams.merge_prob,
                  prior=self.prior,
                  covs=self.covs,
                  n_merge=2,#self.hparams.n_merge,
                  origin_cardinality=self.origin_cardinality
              ) 
                #if len(mus_to_merge) > 0 or len(clusters_to_suppress) > 0: # Original
                if (len(mus_to_merge) > 0 or len(clusters_to_suppress) > 0) and not self.hparams.nomerge:
                    # There are clusters to merge or suppress
                    self.merge_performed = True              
                    self.perform_merge(mus_to_merge, highest_ll_mus, clusters_to_suppress)
            
            if self.debug is not None:
               if self.current_epoch%self.debug==0:
                 self.debug_plot()
            # compute nmi, unique z, etc.
            if self.hparams.log_metrics_at_train and self.hparams.evaluate_every_n_epochs > 0 and self.current_epoch % self.hparams.evaluate_every_n_epochs == 0:
                self.log_clustering_metrics()
            with torch.no_grad():
                if self.hparams.log_emb == "every_n_epochs" and (self.current_epoch % self.hparams.log_emb_every == 0 or self.current_epoch < 2):
                    #self.plot_histograms()
                    #self.plot_utils.visualize_embeddings(
                    #    self.hparams,
                    #    self.logger,
                    #    self.codes_dim,
                    #    vae_means=self.codes,
                    #    vae_labels=None if not self.hparams.use_labels_for_eval else self.train_gt,
                    #    val_resp=self.train_resp,
                    #    current_epoch=self.current_epoch,
                    #    y_hat=None,
                    #    centers=self.mus,
                    #    training_stage='train'
                    #)
                    if self.hparams.dataset == "synthetic":
                        if self.split_performed or self.merge_performed:
                            self.plot_utils.update_colors(self.split_performed, self.mus_ind_to_split, self.mus_inds_to_merge)
                        elif self.hparams.use_labels_for_eval:
                            self.plot_utils.plot_cluster_and_decision_boundaries(samples=self.codes, labels=self.train_resp.argmax(-1), gt_labels=self.train_gt, net_centers=self.mus, net_covs=self.covs, n_epoch=self.current_epoch, cluster_net=self)
                    if self.current_epoch in (0, 1, 2, 3, 4, 5, 10, 100, 200, 300, 400, 500, 549, self.hparams.start_sub_clustering, self.hparams.start_sub_clustering+1) or self.split_performed or self.merge_performed:
                        self.plot_histograms(for_thesis=True)
        
        if self.split_performed or self.merge_performed:
            print('N_SUB_LIST UPDATE PARAMS :',self.n_sub_list)
            self.update_params_split_merge()
            print("Current number of clusters: ", self.K)
        self.log_clustering_params()
        
        
        
        
    
    def debug_plot(self):
        torch.save(self.codes,f'/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/assignement_and_data/codes_{self.current_epoch}.pt')
        torch.save(self.train_gt,f'/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/assignement_and_data/train_gt_{self.current_epoch}.pt')
        torch.save(self.train_resp,f'/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/assignement_and_data/train_resp_{self.current_epoch}.pt')
        torch.save(self.train_resp_sub,f'/home/exx/Documents/Cyril_Kana_Python_Project/DeepDPTGMM/debug/assignement_and_data/train_resp_sub_{self.current_epoch}.pt')
        
    
    def validation_epoch_end(self, outputs):
        # Take mean of all batch losses
        #print('INTO VALIDATION EPOCH END')
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("cluster_net_train/val/avg_val_loss", avg_loss)
        if self.current_training_stage != "gather_codes" and self.hparams.evaluate_every_n_epochs and self.current_epoch % self.hparams.evaluate_every_n_epochs == 0:
            z = self.val_resp.argmax(axis=1).cpu()
            nmi = normalized_mutual_info_score(
                # curr_clusters_assign,
                self.val_gt,
                z,
            )
            self.last_val_NMI = nmi
            self.log_clustering_metrics(stage="val")
            print('INTO VALIDATION EPOCH END')
            if not (self.split_performed or self.merge_performed) and self.hparams.log_metrics_at_train:               
                self.log_clustering_metrics(stage="total")

        if self.hparams.log_emb == "every_n_epochs" and self.current_epoch % 10 == 0 and len(self.val_gt) > 10:
            # not mock data
            self.plot_utils.visualize_embeddings(
                self.hparams,
                self.logger,
                self.codes_dim,
                vae_means=self.codes,
                vae_labels=self.val_gt,
                val_resp=self.val_resp if self.val_resp != [] else None,
                current_epoch=self.current_epoch,
                y_hat=None,
                centers=None,
                training_stage="val_thesis"
            )

        if self.current_epoch > self.hparams.start_sub_clustering and (self.current_epoch % 50 == 0 or self.current_epoch == self.hparams.train_cluster_net - 1):
            from pytorch_lightning.loggers.base import DummyLogger
            if not isinstance(self.logger, DummyLogger):
                self.plot_histograms(train=False, for_thesis=True)
    """
    #To delete 
    def on_after_backward(self):
      # This runs after loss.backward()
      if self.subclustering_net is not None:
        for name, param in self.subclustering_net.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_max = grad.abs().max().item()
                grad_mean = grad.abs().mean().item()
                grad_min = grad.abs().min().item()
    
                # Log statistics
                self.log_dict({
                    f"grad/{name}_max": grad_max,
                    f"grad/{name}_mean": grad_mean,
                    f"grad/{name}_min": grad_min,
                }, on_step=True, on_epoch=False)
    
                # Optionally print alerts
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    print(f"[!!!] NaN or Inf in gradient of {name}")
                elif grad_max > 1e3:
                    print(f"[!!!] Exploding gradient in {name} ? max |grad| = {grad_max:.3e}")
    """

    def subcluster(self, codes, logits, n_sub_list, hard_assignment=True, batch_idx=None, total_batches=None,debug=False):
        #A suppr
        """
        for name, param in self.subclustering_net.named_parameters():
          if param.requires_grad:
              if torch.isnan(param).any() or torch.isinf(param).any():
                  raise RuntimeError(f"ERROR: Parameter `{name}` is already NaN/Inf before forward")
        """
        # Cluster codes into subclusters
        sub_clus_resp = self.subclustering_net(codes)  # Unnormalized scores
        #if batch_idx == total_batches - 1:
        # Check for NaNs in codes
        if torch.isnan(codes).any():
            print("NaN detected in 'codes' before entering subclustering_net")
        if not torch.isfinite(codes).all():
            print("Non-finite values detected in 'codes' before entering subclustering_net")
        if torch.isnan(sub_clus_resp).any():
            print('covs sub :',self.covs_sub)
            print('mus sub :',self.mus_sub)
            print(f"NaN detected in outputs at batch {batch_idx}")
        if torch.isinf(sub_clus_resp).any():
            print(f"Inf detected in outputs at batch {batch_idx}")
          
        z = logits.argmax(-1)  # Get the main cluster assignments
    
        # Initialize the mask with zeros
        mask = torch.zeros_like(sub_clus_resp)
       
        # Compute the starting index for each cluster's subclusters
        cluster_offsets = np.cumsum([0] + n_sub_list[:-1])
        #print('INTO SUBCLUSTER n_sub_list',n_sub_list)
        #print('logits size :',logits.size())
        # Debug: Print cluster offsets
        #print(f"cluster_offsets: {cluster_offsets}")
        # Apply mask based on dynamic number of subclusters per main cluster
        for i, cluster_idx in enumerate(z):
            #print('CLUSTER IDX :',cluster_idx)
            num_subclusters = n_sub_list[cluster_idx]
            cluster_offset = cluster_offsets[cluster_idx]
            # Mask out irrelevant subclusters, based on number of subclusters per main cluster
            mask[i, cluster_offset: cluster_offset + num_subclusters] = 1.0
    
        # Perform softmax, ensuring irrelevant subclusters are excluded
        masked_sub_clus_resp = sub_clus_resp.masked_fill((1 - mask).bool(), float('-inf')) * self.subclustering_net.softmax_norm
        sub_clus_resp = torch.nn.functional.softmax(masked_sub_clus_resp, dim=1)
        if torch.isnan(sub_clus_resp).any():
          print(f"NaN detected in sub_clus_resp at batch {batch_idx}")
          
        # Debug printing at the end of an epoch
        """
        if batch_idx == total_batches - 1:
            print("Method: subcluster")
            for i in range(min(5, len(z))):  # Print for the first 5 data points
                cluster_idx = z[i].item()
                cluster_label = self.cluster_labels[cluster_idx] if self.cluster_labels else f"Cluster {cluster_idx}"
                num_subclusters = n_sub_list[cluster_idx]
                cluster_offset = cluster_offsets[cluster_idx]
                subcluster_indices = list(range(cluster_offset, cluster_offset + num_subclusters))
                subcluster_labels = [
                    self.subcluster_labels[cluster_idx][j] if self.subcluster_labels else f"Subcluster {j}"
                    for j in range(num_subclusters)
                ]
                print(
                    f"Data point {i}: assigned to cluster '{cluster_label}' (MLP index {cluster_idx}), "
                    f"using subclusters {subcluster_labels} (subcluster indices {subcluster_indices})"
                )
        """
        if debug:
          #print('sub_clus_resp.size() :',sub_clus_resp.size())
          return sub_clus_resp,masked_sub_clus_resp 
        else:
          return sub_clus_resp



    """
    def subcluster(self, codes, logits, n_sub_list, hard_assignment=True):
        # Cluster codes into subclusters
        sub_clus_resp = self.subclustering_net(codes)  # Unnormalized scores
        z = logits.argmax(-1)  # Get the main cluster assignments
    
        # Initialize the mask with zeros
        mask = torch.zeros_like(sub_clus_resp)
        
        # Compute the starting index for each cluster's subclusters
        cluster_offsets = np.cumsum([0] + n_sub_list[:-1])
    
        # Apply mask based on dynamic number of subclusters per main cluster
        for i, cluster_idx in enumerate(z):
            num_subclusters = n_sub_list[cluster_idx]
            cluster_offset = cluster_offsets[cluster_idx]
            # Mask out irrelevant subclusters, based on number of subclusters per main cluster
            mask[i, cluster_offset: cluster_offset + num_subclusters] = 1.0
    
        # Perform softmax, ensuring irrelevant subclusters are excluded
        sub_clus_resp = torch.nn.functional.softmax(
            sub_clus_resp.masked_fill((1 - mask).bool(), float('-inf')) * self.subclustering_net.softmax_norm,
            dim=1
        )
        return sub_clus_resp
    """


    
    def subcluster_DPM(self, codes, logits, hard_assignment=True):
        # cluster codes into subclusters
        sub_clus_resp = self.subclustering_net(codes)  # unnormalized
        z = logits.argmax(-1)

        # zero out irrelevant subclusters
        mask = torch.zeros_like(sub_clus_resp)
        mask[np.arange(len(z)), 2 * z] = 1.
        mask[np.arange(len(z)), 2 * z + 1] = 1.

        # perform softmax
        sub_clus_resp = torch.nn.functional.softmax(sub_clus_resp.masked_fill((1 - mask).bool(), float('-inf')) * self.subclustering_net.softmax_norm, dim=1)
        return sub_clus_resp
    def update_subcluster_net_split(self, split_decisions, n_sub_list_new, accepted_subclusters=None):
        """
        Updates the subclustering network after clusters have been split, now supporting partial acceptance.
        """
        subclus_opt = self.optimizers()[self.optimizers_dict_idx["subcluster_net_opt"]]
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p, None)
    
        # The subclustering net also needs to do partial acceptance
        self.subclustering_net.update_K_split(
            split_decisions=split_decisions,
            init_new_weights=self.hparams.split_init_weights_sub,
            n_sub_list_new=n_sub_list_new,
            accepted_subclusters=accepted_subclusters
        )
    
        # Reassign the updated parameters to the optimizer
        subclus_opt.param_groups[0]["params"] = list(self.subclustering_net.parameters())
        self.subclustering_net.to(self._device)
    def update_subcluster_net_split_N_subcluster(self, split_decisions, n_sub_list_new):
        """
        Updates the subclustering network after clusters have been split, handling variable numbers of subclusters per cluster.
    
        Parameters:
        - split_decisions: Tensor of booleans indicating which clusters to split.
        - n_sub_list_new: List containing the updated number of subclusters per cluster after splitting.
        """
        # Update the subcluster net to have the new total number of subclusters
        subclus_opt = self.optimizers()[self.optimizers_dict_idx["subcluster_net_opt"]]
        print('SUBCLUS_OPT :',subclus_opt)
        # Print optimizer state and parameter sizes BEFORE updating
        print('SUBCLUS_OPT BEFORE:', subclus_opt)
        print('subclus_opt.param_groups[0]["params"] BEFORE:',subclus_opt.param_groups[0]["params"])
        for p in subclus_opt.param_groups[0]["params"]:
            print(f"Tensor size: {p.size()}")
        
        # Remove old weights from the optimizer state
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p, None)
       
        # Update the subclustering network with the new clusters and subclusters
        self.subclustering_net.update_K_split(split_decisions, self.hparams.split_init_weights_sub, n_sub_list_new)

        # Print updated parameter sizes AFTER the subclustering network is updated
        print('subclus_opt.param_groups[0]["params"] AFTER:',list(self.subclustering_net.parameters()))
        for p in self.subclustering_net.parameters():
            print(f"Tensor size: {p.size()}")
        subclus_opt.param_groups[0]["params"] = list(self.subclustering_net.parameters())
        self.subclustering_net.to(self._device)

    
    def perform_split_operations(self, split_decisions, accepted_subclusters):
        """
        Handle updates needed when certain clusters are split. 
    
        Parameters:
        -----------
        split_decisions : torch.BoolTensor
            Boolean mask indicating which main clusters are split.
        accepted_subclusters : List[List[int]]
            For each main cluster k, a list of subcluster indices that should become main clusters.
        """
        print("perform_split_operations function clusternetasmodel:", self.current_epoch)
    
        if not self.hparams.ignore_subclusters:
            clus_opt = self.optimizers()[self.optimizers_dict_idx["cluster_net_opt"]]
        else:
            # Only one optimizer
            clus_opt = self.optimizers()
    
        # Remove old weights from the optimizer state (required because we are about to re-init the layer)
        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p, None)  # Use pop with default to avoid KeyError
    
        # Update the cluster network with the new clusters
        # Pass both split_decisions and accepted_subclusters 
        self.cluster_net.update_K_split(
            split_decisions=split_decisions,
            accepted_subclusters=accepted_subclusters,
            init_new_weights=self.hparams.init_new_weights,
            subclusters_nets=self.subclustering_net,
            n_sub_list=self.n_sub_list
        )
        # Update the optimizer's parameter groups with the new parameters
        clus_opt.param_groups[1]["params"] = list(self.cluster_net.class_fc2.parameters())
        self.cluster_net.class_fc2.to(self._device)
    
        # Update model parameters after split
        (
        self.mus_new,
        self.covs_new,
        self.pi_new,
        self.mus_sub_new,
        self.covs_sub_new,
        self.pi_sub_new,
        self.prior,
        n_sub_list_new,
        self.cluster_labels,
        self.subcluster_labels,
        self.prior_labels,
        self.origin_cardinality  # <-- new
    ) = update_models_parameters_split(
        split_decisions,
        self.mus,
        self.covs,
        self.pi,
        self.mus_sub,
        self.covs_sub,
        self.pi_sub,
        self.codes,
        self.train_resp,
        self.train_resp_sub,
        self.n_sub_list,
        self.n_sub,
        self.hparams.how_to_init_mu_sub,
        self.prior,
        use_priors=self.hparams.use_priors,
        cluster_labels=self.cluster_labels,
        subcluster_labels=self.subcluster_labels,
        prior_labels=self.prior_labels,
        accepted_subclusters=accepted_subclusters  # partial subcluster info
    )
        print('ORIGIN CARDINALITY :',self.origin_cardinality)
        print('self.n_sub_list : ',self.n_sub_list)
        # 1) Recalculate how many total new clusters were formed
        num_splits = split_decisions.sum().item()  
        num_not_split = len(split_decisions) - num_splits
        
        split_indices = split_decisions.nonzero(as_tuple=False).squeeze()
        if split_indices.dim() == 0:
            split_indices = split_indices.unsqueeze(0)
        split_indices = split_indices.tolist()
        if isinstance(split_indices, int):
            split_indices = [split_indices]
        
        total_new_clusters = 0
        for k in split_indices:
            # PARTIAL acceptance: only count the accepted subclusters
            total_new_clusters += len(accepted_subclusters[k])
        
        # 2) Update self.K
        self.K = num_not_split + total_new_clusters
        
        print('SELF.N_SUB_LIST before split:', self.n_sub_list)
        print('SELF.n_sub_list_new:', n_sub_list_new)
        print('split_decisions:', split_decisions)
        self.n_sub_list = n_sub_list_new  # Must reflect partial acceptance
        
        # 3) Update the subcluster network if needed
        if not self.hparams.ignore_subclusters:
            # Pass accepted_subclusters along if you need partial acceptance logic inside
            self.update_subcluster_net_split(
                split_decisions, 
                n_sub_list_new, 
                accepted_subclusters=accepted_subclusters
            )
        
        # Record which clusters were split
        self.mus_ind_to_split = split_indices

    def perform_split_operations_N_subcluster(self, split_decisions):
        """
        Generalized perform_split_operations function to handle variable number of subclusters per cluster.
    
        Parameters:
        - split_decisions: Tensor of booleans indicating whether to split each cluster.
        """
        # Update the cluster net to have the new K
        print("perform_split_operations function clusternetasmodel:", self.current_epoch)
    
        if not self.hparams.ignore_subclusters:
            clus_opt = self.optimizers()[self.optimizers_dict_idx["cluster_net_opt"]]
        else:
            # Only one optimizer
            clus_opt = self.optimizers()
    
        # Remove old weights from the optimizer state
        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p, None)  # Use pop with default to avoid KeyError
    
        # Update the cluster network with the new clusters
        self.cluster_net.update_K_split(
            split_decisions, self.hparams.init_new_weights, self.subclustering_net, self.n_sub_list
        )
    
        # Update the optimizer's parameter groups with the new parameters
        clus_opt.param_groups[1]["params"] = list(self.cluster_net.class_fc2.parameters())
        self.cluster_net.class_fc2.to(self._device)
    
        # Update model parameters after split
        (
        self.mus_new,
        self.covs_new,
        self.pi_new,
        self.mus_sub_new,
        self.covs_sub_new,
        self.pi_sub_new,
        self.prior,
        n_sub_list_new,  # Updated n_sub_list
        self.cluster_labels,
        self.subcluster_labels,
        self.prior_labels ) = update_models_parameters_split(
        split_decisions,
        self.mus,
        self.covs,
        self.pi,
        self.mus_sub,
        self.covs_sub,
        self.pi_sub,
        self.codes,
        self.train_resp,
        self.train_resp_sub,
        self.n_sub_list,
        self.n_sub,
        self.hparams.how_to_init_mu_sub,
        self.prior,
        use_priors=self.hparams.use_priors,
        cluster_labels=self.cluster_labels,
        subcluster_labels=self.subcluster_labels,
        prior_labels=self.prior_labels)

        
    
        # Update the number of clusters K
        # Compute the new number of clusters after splitting
        num_splits = split_decisions.sum().item()  # Number of clusters being split
    
        # Number of clusters not being split
        num_not_split = len(split_decisions) - num_splits
    
        # Calculate total new clusters added from splits
        # For each cluster being split, the number of new clusters added is equal to the number of subclusters for that cluster
        #split_indices = split_decisions.nonzero(as_tuple=False).squeeze().tolist()
        split_indices = split_decisions.nonzero(as_tuple=False).squeeze()
        if split_indices.ndim ==0:
          split_indices=torch.tensor([split_indices.item()])
        split_indices=split_indices.tolist()
        
        if isinstance(split_indices, int):
            split_indices = [split_indices]
    
        total_new_clusters = 0
        for k in split_indices:
            num_subclusters_k = self.n_sub_list[k]
            total_new_clusters += num_subclusters_k
    
        # Update self.K
        self.K = num_not_split + total_new_clusters
    
        # Update n_sub_list with the new subcluster counts
        print('SELF.N_SUB_LIST before split:',self.n_sub_list)
        print('SELF.n_sub_list_new :',n_sub_list_new)
        print('split_decisions :',split_decisions)
        self.n_sub_list = n_sub_list_new
    
        if not self.hparams.ignore_subclusters:
            # Update subcluster network
            self.update_subcluster_net_split(split_decisions, n_sub_list_new)
    
        # Record which clusters were split
        #self.mus_ind_to_split = split_decisions.nonzero(as_tuple=False).squeeze()
        self.mus_ind_to_split=split_indices


    def perform_split_operations_2sub(self, split_decisions):
        # split_decisions is a list of k boolean indicators of whether we would want to split cluster k
        # update the cluster net to have the new K
        print("perform_split_operations function clusternetasmodel :",self.current_epoch)
        if not self.hparams.ignore_subclusters:
            clus_opt = self.optimizers()[self.optimizers_dict_idx["cluster_net_opt"]]
        else:
            # only one optimizer
            clus_opt = self.optimizers()

        # remove old weights from the optimizer state
        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p)
        self.cluster_net.update_K_split(
            split_decisions, self.hparams.init_new_weights, self.subclustering_net
        )
        clus_opt.param_groups[1]["params"] = list(self.cluster_net.class_fc2.parameters())
        self.cluster_net.class_fc2.to(self._device)
        mus_ind_to_split = torch.nonzero(torch.tensor(split_decisions), as_tuple=False)
        (
            self.mus_new,
            self.covs_new,
            self.pi_new,
            self.mus_sub_new,
            self.covs_sub_new,
            self.pi_sub_new,
            self.prior
        ) = update_models_parameters_split(
            split_decisions,
            self.mus,
            self.covs,
            self.pi,
            mus_ind_to_split,
            self.mus_sub,
            self.covs_sub,
            self.pi_sub,
            self.codes,
            self.train_resp,
            self.train_resp_sub,
            self.n_sub,
            self.hparams.how_to_init_mu_sub,
            self.prior,
            use_priors=self.hparams.use_priors
        )
        # update K
        #print(f"Splitting clusters {np.arange(self.K)[split_decisions.bool().tolist()]}")
        self.K += len(mus_ind_to_split)

        if not self.hparams.ignore_subclusters:
            # update subclusters_net
            self.update_subcluster_net_split(split_decisions)
        self.mus_ind_to_split = mus_ind_to_split
    
    
    def update_subcluster_nets_merge(self, inds_to_mask, mus_lists_to_merge, highest_ll_mus, clusters_to_suppress):
        """
        Updates the subclustering network after clusters have been merged and suppressed.
    
        Args:
            inds_to_mask (torch.Tensor): Boolean tensor indicating clusters to be merged or suppressed.
            mus_lists_to_merge (list): List of lists containing indices of clusters to merge.
            highest_ll_mus (list): Indices of clusters with the highest log-likelihood in each merge group.
            clusters_to_suppress (list): Indices of clusters to suppress (remove).
        """
        # Combine inds_to_mask and clusters_to_suppress
        total_inds_to_mask = inds_to_mask.clone()
        for idx in clusters_to_suppress:
            total_inds_to_mask[idx] = True
    
        # Update the cluster net to have the new K
        subclus_opt = self.optimizers()[self.optimizers_dict_idx["subcluster_net_opt"]]
    
        # Remove old weights from the optimizer state
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p, None)  # Use pop with default to avoid KeyError
    
        # Update the subclustering network
        self.subclustering_net.update_K_merge(
            total_inds_to_mask,
            mus_lists_to_merge=mus_lists_to_merge,
            highest_ll_mus=highest_ll_mus,
            clusters_to_suppress=clusters_to_suppress,
            init_new_weights=self.hparams.merge_init_weights_sub,
            n_sub_list_new=self.n_sub_list  # Updated n_sub_list after merge
        )
    
        subclus_opt.param_groups[0]["params"] = list(self.subclustering_net.parameters())


    def perform_merge(self, mus_lists_to_merge, highest_ll_mus, clusters_to_suppress, use_priors=True):
        """
        Performs merges and suppressions of clusters.
    
        Args:
            mus_lists_to_merge (list): List of lists containing indices of clusters to merge.
            highest_ll_mus (list): Indices of clusters with the highest log-likelihood in each merge group.
            clusters_to_suppress (list): Indices of clusters to suppress (remove).
        """
        print(f"Merging clusters {mus_lists_to_merge}")
        print(f"Suppressing clusters {clusters_to_suppress}")
    
        # Create a boolean mask of clusters to be merged or suppressed
        inds_to_mask = torch.zeros(self.K, dtype=bool)
        for group in mus_lists_to_merge:
            inds_to_mask[group] = True
        for idx in clusters_to_suppress:
            inds_to_mask[idx] = True
    
        # Update model parameters to reflect merges and suppressions
        (
            self.mus_new,
            self.covs_new,
            self.pi_new,
            self.mus_sub_new,
            self.covs_sub_new,
            self.pi_sub_new,
            self.prior,
            self.n_sub_list,
        ) = update_models_parameters_merge(
            mus_lists_to_merge,
            inds_to_mask,
            self.K,
            self.mus,
            self.covs,
            self.pi,
            self.mus_sub,
            self.covs_sub,
            self.pi_sub,
            self.codes,
            self.train_resp,
            self.prior,
            use_priors=self.hparams.use_priors,
            n_sub=self.n_sub,
            how_to_init_mu_sub=self.hparams.how_to_init_mu_sub,
            n_sub_list=self.n_sub_list,
            clusters_to_suppress=clusters_to_suppress  # Pass suppressed clusters
        )
    
        # Adjust K
        num_clusters_merged = sum(len(group) for group in mus_lists_to_merge)
        num_new_clusters = len(mus_lists_to_merge)
        num_clusters_suppressed = len(clusters_to_suppress)
        self.K = self.K - num_clusters_merged + num_new_clusters - num_clusters_suppressed
    
        print(f"New number of clusters K: {self.K}")
    
        # Update subclustering net if applicable
        if not self.hparams.ignore_subclusters:
            self.update_subcluster_nets_merge(inds_to_mask, mus_lists_to_merge, highest_ll_mus, clusters_to_suppress)
    
        # Update the cluster net to have the new K
        if not self.hparams.ignore_subclusters:
            clus_opt = self.optimizers()[self.optimizers_dict_idx["cluster_net_opt"]]
        else:
            # Only one optimizer
            clus_opt = self.optimizers()
    
        # Remove old weights from the optimizer state
        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p, None)
    
        # Update cluster net
        self.cluster_net.update_K_merge(
            inds_to_mask,
            mus_lists_to_merge=mus_lists_to_merge,
            highest_ll_mus=highest_ll_mus,
            clusters_to_suppress=clusters_to_suppress,
            init_new_weights=self.hparams.init_new_weights,
        )
        # Add parameters to the optimizer
        clus_opt.param_groups[1]["params"] = list(self.cluster_net.class_fc2.parameters())
    
        self.cluster_net.class_fc2.to(self._device)
        self.mus_inds_to_merge = mus_lists_to_merge
        #self.merge_not_performed=False

    
    def configure_optimizers(self):
        # Get all params except for the last layer
        cluster_params = torch.nn.ParameterList([p for n, p in self.cluster_net.named_parameters() if "class_fc2" not in n])
        
        # Optimizer for the main cluster network
        cluster_net_opt = optim.Adam(cluster_params, lr=self.hparams.cluster_lr)
        
        # Adding distinct parameter group for the last layer for easier updates
        cluster_net_opt.add_param_group(
            {"params": self.cluster_net.class_fc2.parameters()}
        )
        
        # Store optimizer index for reference
        self.optimizers_dict_idx = {
            "cluster_net_opt": 0
        }
    
        # Setup the learning rate scheduler if necessary
        if self.hparams.lr_scheduler == "StepLR":
            cluster_scheduler = torch.optim.lr_scheduler.StepLR(cluster_net_opt, step_size=20)
            print("StepLR")
        elif self.hparams.lr_scheduler == "ReduceOnP":
            cluster_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cluster_net_opt, mode="min", factor=0.5, patience=4)
            print("ReduceOnP")
        else:
            cluster_scheduler = None
            print('NO OPTIMIZER')
    
        # Initialize subclustering optimizer with a dummy parameter (instead of real subclustering_net params)
        self.dummy_param = nn.Parameter(torch.zeros(1))  # Dummy parameter for now
        sub_clus_opt = optim.Adam([self.dummy_param], lr=self.hparams.subcluster_lr)
    
        # Store the subclustering optimizer index
        self.optimizers_dict_idx["subcluster_net_opt"] = 1
    
        # Return the cluster optimizer, scheduler, and subclustering optimizer
        return (
        [{"optimizer": cluster_net_opt, "scheduler": cluster_scheduler, "monitor": "cluster_net_train/val/cluster_loss"},
         {"optimizer": sub_clus_opt}]
        if cluster_scheduler else [cluster_net_opt, sub_clus_opt]
    )


    def configure_optimizers_DPM(self):
        #print('into_configure_optimizer')
        # Get all params but last layer
        cluster_params = torch.nn.ParameterList([p for n, p in self.cluster_net.named_parameters() if "class_fc2" not in n])
        cluster_net_opt = optim.Adam(
            cluster_params, lr=self.hparams.cluster_lr
        )
        # distinct parameter group for the last layer for easy update
        cluster_net_opt.add_param_group(
            {"params": self.cluster_net.class_fc2.parameters()}
        )
        self.optimizers_dict_idx = {
            "cluster_net_opt": 0
        }

        if self.hparams.lr_scheduler == "StepLR":
            cluster_scheduler = torch.optim.lr_scheduler.StepLR(cluster_net_opt, step_size=20)
            print("StepLR")
        elif self.hparams.lr_scheduler == "ReduceOnP":
            cluster_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cluster_net_opt, mode="min", factor=0.5, patience=4)
            print("ReduceOnP")
        else:
            cluster_scheduler = None
            print('NO OPTIMIZER')

        if not self.hparams.ignore_subclusters:
            sub_clus_opt = optim.Adam(
                self.subclustering_net.parameters(), lr=self.hparams.subcluster_lr
            )
            self.optimizers_dict_idx["subcluster_net_opt"] = 1
            #print('AAAAAAA')
            return (
                {"optimizer": cluster_net_opt, "scheduler": cluster_scheduler, "monitor": "cluster_net_train/val/cluster_loss"},
                {"optimizer": sub_clus_opt, }
            )
        return {"optimizer": cluster_net_opt, "scheduler": cluster_scheduler, "monitor": "cluster_net_train/val/cluster_loss"} if cluster_scheduler else cluster_net_opt

    def update_params_split_merge(self):
        self.mus = self.mus_new
        self.covs = self.covs_new
        self.mus_sub = self.mus_sub_new
        self.covs_sub = self.covs_sub_new
        self.pi = self.pi_new
        self.pi_sub = self.pi_sub_new

    def init_covs_and_pis_given_mus(self):
        # each point will be hard assigned to its closest cluster and then compute covs and pis.
        # compute dist mat
        print('INTO reinit_params_given_mus ')
        print('self.hparams.use_priors_for_net_params_init :',self.hparams.use_priors_for_net_params_init)
        print('self.hparams.reinit_params_given_mus ',self.hparams.reinit_params_given_mus)
        if self.hparams.use_priors_for_net_params_init:
            _, cov_prior = self.prior.init_priors(self.mus)  # giving mus and nopt codes because we only need the dim for the covs
            self.covs = torch.stack([cov_prior for k in range(self.K)])
            p_counts = torch.ones(self.K) * 10
            self.pi = p_counts / float(self.K * 10)  # a uniform pi prior

        else:
            if self.hparams.reinit_params_given_mus == "min_dist":
              dis_mat = torch.empty((len(self.codes), self.K))
              for i in range(self.K):
                  # Compute cosine similarity
                  dot_product = (self.codes * self.mus[i]).sum(axis=1)
                  norm_codes = torch.norm(self.codes, dim=1)
                  norm_mus = torch.norm(self.mus[i], dim=0)
                  cosine_similarity = dot_product / (norm_codes * norm_mus + 1e-8)  # Add epsilon to prevent division by zero
                  # Convert cosine similarity to cosine dissimilarity
                  cosine_dissimilarity = 1 - cosine_similarity
                  dis_mat[:, i] = cosine_dissimilarity
              
              # Get hard assignment
              hard_assign = torch.argmin(dis_mat, dim=1)
    
              
            if self.hparams.reinit_params_given_mus == "umap":
                print('INTO reinit_params_given_mus UMAP')
                # Project codes and mus into UMAP space using cosine distance
                umap_obj = umap.UMAP(
                    n_neighbors=30,
                    min_dist=0.1,
                    n_components=3,
                    random_state=42,
                    metric='cosine'
                )
                
                # Move tensors to CPU and convert to NumPy for UMAP
                codes_np = self.codes.detach().cpu().numpy()
                mus_np = self.mus.detach().cpu().numpy()
                
                # Concatenate codes and mus for joint embedding
                combined = np.vstack((codes_np, mus_np))
                
                # Fit UMAP on combined data
                umap_obj.fit(combined)
                embedding = umap_obj.embedding_
                
                # Separate the embeddings
                umap_codes = embedding[:len(codes_np)]
                umap_mus = embedding[len(codes_np):]
                
                # Convert embeddings back to torch tensors
                umap_codes_tensor = torch.from_numpy(umap_codes).to(self.codes.device)
                umap_mus_tensor = torch.from_numpy(umap_mus).to(self.codes.device)
                
                # Compute pairwise L2 distances between codes and mus in UMAP space
                # Efficient computation using broadcasting
                # umap_codes_tensor: [N, D], umap_mus_tensor: [K, D]
                # Resulting distance matrix: [N, K]
                diff = umap_codes_tensor.unsqueeze(1) - umap_mus_tensor.unsqueeze(0)  # [N, K, D]
                l2_distances = torch.norm(diff, dim=2)  # [N, K]
                
                # Get hard assignment based on minimum L2 distance
                hard_assign = torch.argmin(l2_distances, dim=1)
            if self.hparams.reinit_params_given_mus == "umap_kmeans_with_mus_center":
                print('INTO reinit_params_given_mus UMAP + KMEANS + KARCHER')
                umap_obj = umap.UMAP(
                    n_neighbors=30,
                    min_dist=0.1,
                    n_components=3,
                    random_state=42,
                    metric='cosine'
                )
                codes_np = self.codes.detach().cpu().numpy()   # [N, D]
                mus_np   = self.mus.detach().cpu().numpy()     # [K, D]
                combined = np.vstack((codes_np, mus_np))       # [N+K, D]
                embedding = umap_obj.fit_transform(combined)   # [N+K, 3]
                umap_codes = embedding[:codes_np.shape[0], :]  # [N, 3]
                umap_mus   = embedding[codes_np.shape[0]:, :]  # [K, 3]
            
                # 2) K-means warm-start with the projected mus
                from sklearn.cluster import KMeans
                K = umap_mus.shape[0]
                kmeans = KMeans(
                    n_clusters=K,
                    init=umap_mus,
                    n_init=1,
                    max_iter=300,
                )
                kmeans.fit(umap_codes)                         # cluster in UMAP space
            
                # 3) hard assignments from K-means
                hard_assign = torch.from_numpy(kmeans.labels_) \
                                   .long() \
                                   .to(self.codes.device)
            
                # 4) recompute self.mus via KarcherMean on the original codes
                #    (assuming KarcherMean returns a tensor on .device by default)
                device = self.codes.device
                labels = hard_assign
                num_clusters = mus_np.shape[0]
                self.mus = torch.stack([
                    KarcherMean(
                        soft_assign=None,
                        codes=self.codes[labels == i]
                    ).to(device=device)
                    for i in range(num_clusters)
                ])
            if self.hparams.reinit_params_given_mus == "kmeansumap_adel":
                print('INTO reinit_params_given_mus UMAP + KMEANS + KARCHER')
                umap_obj = umap.UMAP(
                    n_neighbors=30,
                    min_dist=0.1,
                    n_components=3,
                    random_state=42,
                    metric='cosine'
                )
                codes_np = self.codes.detach().cpu().numpy()   # [N, D]
                mus_np   = self.mus.detach().cpu().numpy()     # [K, D]
                combined = np.vstack((codes_np, mus_np))       # [N+K, D]
                embedding = umap_obj.fit_transform(combined)   # [N+K, 3]
                umap_codes = embedding[:codes_np.shape[0], :]  # [N, 3]
                umap_mus   = embedding[codes_np.shape[0]:, :]  # [K, 3]
            
                # 2) K-means warm-start with the projected mus
                from sklearn.cluster import KMeans
                K = umap_mus.shape[0]
                kmeans = KMeans(
                    n_clusters=K,
                    n_init=1,
                    max_iter=300,
                )
                kmeans.fit(umap_codes)                         # cluster in UMAP space
            
                # 3) hard assignments from K-means
                hard_assign = torch.from_numpy(kmeans.labels_) \
                                   .long() \
                                   .to(self.codes.device)
            
                # 4) recompute self.mus via KarcherMean on the original codes
                #    (assuming KarcherMean returns a tensor on .device by default)
                device = self.codes.device
                labels = hard_assign
                num_clusters = mus_np.shape[0]
                self.mus = torch.stack([
                    KarcherMean(
                        soft_assign=None,
                        codes=self.codes[labels == i]
                    ).to(device=device)
                    for i in range(num_clusters)
                ])
            # data params
            vals, counts = torch.unique(hard_assign, return_counts=True)
            if len(counts) < self.K:
                new_counts = []
                for k in range(self.K):
                    if k in vals:
                        new_counts.append(counts[vals == k])
                    else:
                        new_counts.append(0)
                counts = torch.tensor(new_counts)
            pi = counts / float(len(self.codes))
            data_covs = compute_data_covs_hard_assignment(hard_assign, self.codes, self.K, self.mus, self.prior)
            if self.use_priors:
                covs = []
                if self.prior_sigma_choice == 'dynamic_data_std':
                  #not implemented yet
                  pass
                  for k in range(self.K):
                      cov_k = self.prior.compute_post_cov(counts[k],data_covs[k],D,psi_index=k)
                      covs.append(cov_k)
                else : 
                   D=len(self.codes[0])
                   for k in range(self.K):
                      cov_k = self.prior.compute_post_cov(counts[k],data_covs[k],D)
                      if not positive_definite.check(cov_k):
                        cov_k=ensure_positive_definite(cov_k)
                      covs.append(cov_k)
                data_covs = torch.stack(covs)
              #print('COVS')
            #else:
            covs = data_covs
            self.covs = covs
            self.pi = pi
        #elif self.hparams.reinit_params_given_mus == "kmeans":
        

    def log_logits(self, logits):
        for k in range(self.K):
            max_k = logits[logits.argmax(axis=1) == k].detach().cpu().numpy()
            if len(max_k > 0):
                fig = plt.figure(figsize=(10, 3))
                for i in range(len(max_k[:20])):
                    if i == 0:
                        plt.bar(np.arange(self.K), max_k[i], fill=False, label=len(max_k))
                    else:
                        plt.bar(np.arange(self.K), max_k[i], fill=False)
                plt.xlabel("Clusters inds")
                plt.ylabel("Softmax histogram")
                plt.title(f"Epoch {self.current_epoch}: cluster {k}")
                plt.legend()

                # self.logger.log_image(f"cluster_net_train/val/logits_reaction_fig_cluster_{k}", fig)
                plt.close(fig)

    def plot_histograms(self, train=True, for_thesis=False):
        pi = self.pi_new if self.split_performed or self.merge_performed else self.pi
        if self.hparams.ignore_subclusters:
            pi_sub = None
        else:
            pi_sub = (
                self.pi_sub_new
                if self.split_performed or self.merge_performed
                else self.pi_sub
                if self.hparams.start_sub_clustering <= self.current_epoch
                else None
            )

        fig = self.plot_utils.plot_weights_histograms(
            K=self.K,
            pi=pi,
            start_sub_clustering=self.hparams.start_sub_clustering,
            current_epoch=self.current_epoch,
            pi_sub=pi_sub,
            for_thesis=for_thesis
        )
        if for_thesis:
            stage = "val_for_thesis"
        else:
            stage = "train" if train else "val"

        from pytorch_lightning.loggers.base import DummyLogger
        if not isinstance(self.logger, DummyLogger):
            self.logger.log_image(f"cluster_net_train/{stage}/clusters_weights_fig", fig)
        plt.close(fig)

    def plot_clusters_high_dim(self, stage="train"):
        resps = {
            "train": (self.train_resp, self.train_resp_sub),
            "val": (self.val_resp, self.val_resp_sub),
        }
        gt = {"train": self.train_gt, "val": self.val_gt}
        (resp, resp_sub) = resps[stage]
        cluster_net_labels = self.training_utils.update_labels_after_split_merge(
            resp.argmax(-1),
            self.split_performed,
            self.merge_performed,
            self.mus,
            self.mus_ind_to_split,
            self.mus_inds_to_merge,
            resp_sub,
        )
        fig = self.plot_utils.plot_clusters_colored_by_label(
            samples=self.codes,
            y_gt=gt[stage],
            n_epoch=self.current_epoch,
            K=len(torch.unique(gt[stage])),
        )
        plt.close(fig)
        self.logger.log_image(f"cluster_net_train/{stage}/clusters_fig_gt_labels", fig)
        fig = self.plot_utils.plot_clusters_colored_by_net(
            samples=self.codes,
            y_net=cluster_net_labels,
            n_epoch=self.current_epoch,
            K=len(torch.unique(cluster_net_labels)),
        )
        self.logger.log_image("cluster_net_train/train/clusters_fig_net_labels", fig)
        plt.close(fig)

    def log_clustering_metrics(self, stage="train"):
        print("Evaluating...")
        if stage == "train":
            gt, resp = self.train_gt, self.train_resp
        elif stage == "val":
            gt, resp = self.val_gt, self.val_resp
            self.log("cluster_net_train/Networks_k", self.K)
        else:  # total
            gt = torch.cat([self.train_gt, self.val_gt])
            resp = torch.cat([self.train_resp, self.val_resp])
        
        # Make sure resp is a tensor before using .argmax
        if isinstance(resp, list):
            resp = torch.cat(resp, dim=0)  # assumes each element is a [B, K] tensor
        
        z = resp.argmax(dim=1).cpu()
        unique_z = len(np.unique(z.numpy()))

    
        if unique_z >= 5:
            _, z_top5 = torch.topk(resp, k=5, largest=True)
        else:
            z_top5 = None
    
        if (gt < 0).any():
            mask = gt > -1
            z = z[mask]
            if z_top5 is not None:
                z_top5 = z_top5[mask]
            gt = gt[mask]
    
        # compute core metrics
        gt_nmi = normalized_mutual_info_score(gt, z)
        ari    = adjusted_rand_score(gt, z)
        acc_top5, acc = training_utils.cluster_acc(gt, z, z_top5)
    
        # log to Lightning’s logger
        self.log(f"cluster_net_train/{stage}/{stage}_nmi",      gt_nmi,   on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_ari",      ari,      on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_acc",      acc,      on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_acc_top5", acc_top5, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/unique_z",         unique_z,on_epoch=True, on_step=False)
    
        log_path = self.hparams.log_perf_cyril
        log_per_epoch_path = self.hparams.get("log_perf_per_epoch", None)
        #print('LOG PATH :',log_path)
        if log_path and log_path != "None":
            has_val = hasattr(self, 'val_gt') and self.val_gt.numel() > 0
            correct_stage = (has_val and stage == "val") or (not has_val and stage == "train")
            if correct_stage:
                # update this run's “best by accuracy”
                if acc > self.run_best_acc:
                    self.run_best_acc = acc
                    self.run_best_nmi = gt_nmi
                    self.run_best_ari = ari
                    self.best_epoch = self.current_epoch
        
                # log best summary at last epoch
                last_epoch = self.hparams.max_epochs - 1
                #print('Last_epoch:', last_epoch)
                if self.current_epoch == last_epoch:
                    import os
                    from datetime import datetime
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_path, "a") as f:
                        f.write(
                            f"{ts}\t"
                            f"best_acc {self.run_best_acc:.4f}\t"
                            f"best_nmi {self.run_best_nmi:.4f}\t"
                            f"best_ari {self.run_best_ari:.4f}\t"
                            f"predicted K {unique_z}\t"
                            f"Epoch {self.best_epoch}\n"
                        )
        
        # === New logging: write per-epoch performance and overwrite at each new run ===
        if log_per_epoch_path and log_per_epoch_path != "None":
            import os
            os.makedirs(os.path.dirname(log_per_epoch_path), exist_ok=True)
        
            # Overwrite at first epoch
            write_mode = "w" if self.current_epoch == 0 else "a"
            with open(log_per_epoch_path, write_mode) as f:
                f.write(
                    f"Epoch {self.current_epoch:03d}\t"
                    f"acc {acc:.4f}\t"
                    f"nmi {gt_nmi:.4f}\t"
                    f"ari {ari:.4f}\t"
                    f"K {unique_z}\n"
                )

    
        # optional console print when offline
        if self.hparams.offline and stage in ("train", "val"):
            print(f"Stage: {stage}, NMI: {gt_nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, K: {self.K}, unique z: {unique_z}")
    
        # alternate logging on special epochs
        if self.current_epoch in (0, 1, self.hparams.train_cluster_net - 1):
            alt_stage = (
                "start"
                if self.current_epoch == 1 or self.hparams.train_cluster_net % self.current_epoch == 0
                else "end"
            )
    
            if unique_z > 1:
                try:
                    silhouette = silhouette_score(self.codes.cpu(), z.cpu().numpy())
                except:
                    silhouette = 0
            else:
                silhouette = 0
    
            ami = adjusted_mutual_info_score(gt.numpy(), z.numpy())
            hom, comp, v_meas = homogeneity_completeness_v_measure(gt.numpy(), z.numpy())
    
            metrics = {
                "nmi": gt_nmi, "ari": ari, "acc": acc, "acc_top5": acc_top5,
                "silhouette_score": silhouette, "ami": ami,
                "homogeneity": hom, "completeness": comp, "v_measure": v_meas,
                "unique_z": unique_z
            }
            for name, val in metrics.items():
                self.log(
                    f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_{name}",
                    val, on_epoch=True, on_step=False
                )


    #### Version sans enregistrer les meilleurs perf pour le papier###
    """
    def log_clustering_metrics(self, stage="train"):
        print("Evaluating...")
        if stage == "train":
            gt = self.train_gt
            resp = self.train_resp
        elif stage == "val":
            gt = self.val_gt
            resp = self.val_resp
            self.log("cluster_net_train/Networks_k", self.K)
        elif stage == "total":
            gt = torch.cat([self.train_gt, self.val_gt])
            resp = torch.cat([self.train_resp, self.val_resp])

        z = resp.argmax(axis=1).cpu()
        unique_z = len(np.unique(z))
        
        if len(np.unique(z)) >= 5:
            val, z_top5 = torch.topk(resp, k=5, largest=True)
        else:
            z_top5 = None
        if (gt < 0).any():
            z = z[gt > -1]
            z_top5 = z_top5[gt > -1]
            gt = gt[gt > -1]

        gt_nmi = normalized_mutual_info_score(gt, z)
        ari = adjusted_rand_score(gt, z)
        acc_top5, acc = training_utils.cluster_acc(gt, z, z_top5)
        
        self.log(f"cluster_net_train/{stage}/{stage}_nmi", gt_nmi, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_ari", ari, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_acc", acc, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_acc_top5", acc_top5, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/unique_z", unique_z, on_epoch=True, on_step=False)

        #if self.hparams.offline and ((self.hparams.log_metrics_at_train and stage == "train") or ( not self.hparams.log_metrics_at_train and stage!="train")):
        if (self.hparams.offline and stage=="val") or(self.hparams.offline and stage=="train"):
            #print(f"nb_z_per_cluster: {np.unique(z,return_counts=True)}")
            print(f"Stage: {stage}, NMI : {gt_nmi}, ARI: {ari}, ACC: {acc}, current K: {self.K} , unique z: {unique_z}")

        if self.current_epoch in (0, 1, self.hparams.train_cluster_net - 1):
            alt_stage = "start" if self.current_epoch == 1 or self.hparams.train_cluster_net % self.current_epoch == 0 else "end"

            if unique_z > 1:
                try:
                    silhouette = silhouette_score(self.codes.cpu(), z.cpu().numpy())
                except:
                    silhouette = 0
            else:
                silhouette = 0
            ami = adjusted_mutual_info_score(gt.numpy(), z.numpy())
            (homogeneity, completeness, v_measure) = homogeneity_completeness_v_measure(gt.numpy(), z.numpy())

            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_nmi", gt_nmi, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_ari", ari, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_acc", acc, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_acc_top5", acc_top5, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_silhouette_score", silhouette, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_ami", ami, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_homogeneity", homogeneity, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_v_measure", v_measure, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_completeness", completeness, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_unique_z", unique_z, on_epoch=True, on_step=False)"""

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        #parser.add_argument("--reinit_with_mus",action="store_true",help='clusternet retraining initialisation  with/without precomputed mus')
        parser.add_argument("--split_merge_mode",
            type=str,
            default="both",
            choices=["both", "split_only", "merge_only"],
            help=(
                "How to use split/merge during clustering: "
                "'both' = alternate split and merge (default), "
                "'split_only' = replace merge phases by split phases, "
                "'merge_only' = replace split phases by merge phases."
            ),
        )
        parser.add_argument(
        "--covariance",
        type=str,
        default="full",
        choices=["iso", "diag", "full"],
        help="choose a covariance constraint during EM inferrence , default is full ",
        )
        ##les --nosplit nomerge nofsub ont été ajouté par moi pour l'ablation study du papier 
        parser.add_argument(
            "--nosplit",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--nomerge",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--nofsub",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--log_perf_cyril",
            type=str,
            default="None",
            help="Path to save the best clustering accuracy, NMI, and ARI for each run"
        )
        parser.add_argument(
            "--log_perf_per_epoch",
            type=str,
            default=None,
            help="If specified, append the  evaluation (ACC,NMI,ARI) each epoch to this text file (be careful it overwrites  path and file name already exist"
        )

        parser.add_argument(
            "--reinit_params_given_mus",
            type=str,
            default="umap",
            choices=["umap","min_dist"],
            help="covs and pis computations given mus at clusternet reinitialisation",
        )
        parser.add_argument(
            "--init_k", default=3, type=int, help="number of initial clusters"
        )
        parser.add_argument("--n_sub",default=2,type=int,help='number of subclusters candidate')
        parser.add_argument("--n_merge",default=2, type=int , help='number of clusters candidate for merging')
        parser.add_argument(
            "--clusternet_hidden",
            type=int,
            default=50,
            help="The dimensions of the hidden dim of the clusternet. Defaults to 50.",
        )
        parser.add_argument(
            "--clusternet_hidden_layer_list",
            type=int,
            nargs="+",
            default=[50],
            help="The hidden layers in the clusternet. Defaults to [50, 50].",
        )
        parser.add_argument(
            "--transform_input_data",
            type=str,
            default="None",
            choices=["normalize", "min_max", "standard", "standard_normalize", "None", None],
            help="Use normalization for embedded data",
        )
        parser.add_argument(
            "--cluster_loss_weight",
            type=float,
            default=1,
        )
        parser.add_argument(
            "--init_cluster_net_weights",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--when_to_compute_mu",
            type=str,
            choices=["once", "every_epoch", "every_5_epochs"],
            default="every_epoch",
        )
        parser.add_argument(
            "--how_to_compute_mu",
            type=str,
            choices=["kmeans", "soft_assign"],
            default="soft_assign",
        )
        parser.add_argument(
            "--how_to_init_mu",
            type=str,
            choices=["kmeans", "soft_assign", "kmeans_1d","umap"],
            default="umap",
        )
        parser.add_argument(
            "--how_to_init_mu_sub",
            type=str,
            choices=["kmeans", "soft_assign", "kmeans_1d","umap"],
            default="umap",
        )
        parser.add_argument(
            "--log_emb_every",
            type=int,
            default=20,
        )
        parser.add_argument(
            "--log_emb",
            type=str,
            default="never",
            choices=["every_n_epochs", "only_sampled", "never"]
        )
        parser.add_argument(
            "--train_cluster_net",
            type=int,
            default=300,
            help="Number of epochs to pretrain the cluster net",
        )
        parser.add_argument(
            "--cluster_lr",
            type=float,
            default=0.0005,
        )
        parser.add_argument(
            "--subcluster_lr",
            type=float,
            default=0.005,
        )
        parser.add_argument(
            "--lr_scheduler", type=str, default="ReduceOnP", choices=["StepLR", "None", "ReduceOnP"]
        )
        parser.add_argument(
            "--start_sub_clustering",
            type=int,
            default=35,
        )
        parser.add_argument(
            "--subcluster_loss_weight",
            type=float,
            default=1.0,
        )
        #start splitting indicate the first splitting analyses , then split_merge_every_n_epochs alternate between merge and split ( pls consider start_merging  also)
        parser.add_argument(
            "--start_splitting",
            type=int,
            default=45,
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=10.0,
        )
        parser.add_argument(
            "--softmax_norm",
            type=float,
            default=1,
        )
        parser.add_argument(
            "--subcluster_softmax_norm",
            type=float,
            default=1,
        )
        parser.add_argument(
            "--split_prob",
            type=float,
            default=None,
            help="Split with this probability even if split rule is not met.  If set to None then the probability that will be used is min(1,H).",
        )
        parser.add_argument(
            "--merge_prob",
            type=float,
            default=None,
            help="merge with this probability even if merge rule is not met. If set to None then the probability that will be used is min(1,H).",
        )
        parser.add_argument(
            "--init_new_weights",
            type=str,
            default="same",
            choices=["same", "random", "subclusters"],
            help="How to create new weights after split. Same duplicates the old cluster's weights to the two new ones, random generate random weights and subclusters copies the weights from the subclustering net",
        )
        parser.add_argument(
            "--start_merging",
            type=int,
            default=45,
            help="The epoch in which to start consider merge proposals",
        )
        parser.add_argument(
            "--merge_init_weights_sub",
            type=str,
            default="highest_ll",
            help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
        )
        parser.add_argument(
            "--split_init_weights_sub",
            type=str,
            default="random",
            choices=["same_w_noise", "same", "random"],
            help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
        )
        parser.add_argument(
            "--split_every_n_epochs",
            type=int,
            default=10,
            help="Example: if set to 10, split proposals will be made every 10 epochs",
        )
        parser.add_argument(
            "--split_merge_every_n_epochs",
            type=int,
            default=30,
            help="Example: if set to 10, split proposals will be made every 10 epochs",
        )
        parser.add_argument(
            "--merge_every_n_epochs",
            type=int,
            default=10,
            help="Example: if set to 10, merge proposals will be made every 10 epochs",
        )
        parser.add_argument(
            "--raise_merge_proposals",
            type=str,
            default="brute_force_NN",
            help="how to raise merge proposals",
        )
        parser.add_argument(
            "--cov_const",
            type=float,
            default=0.005,
            help="gmms covs (in the Hastings ratio) will be torch.eye * cov_const",
        )
        parser.add_argument(
            "--freeze_mus_submus_after_splitmerge",
            type=int,
            default=2,
            help="Numbers of epochs to freeze the mus and sub mus following a split or a merge step",
        )
        parser.add_argument(
            "--freeze_mus_after_init",
            type=int,
            default=5,
            help="Numbers of epochs to freeze the mus and sub mus following a new initialization",
        )
        parser.add_argument(
            "--use_priors",
            type=int,
            default=1,
            help="Whether to use priors when computing model's parameters",
        )
        parser.add_argument("--prior", type=str, default="NIW", choices=["NIW", "NIG"])
        parser.add_argument(
            "--pi_prior", type=str, default="uniform", choices=["uniform", None]
        )
        parser.add_argument(
            "--prior_dir_counts",
            type=float,
            default=0.1,
        )
        parser.add_argument(
            "--prior_kappa",
            type=float,
            default=0.0001,
        )
        parser.add_argument(
            "--NIW_prior_nu",
            type=float,
            default=None,
            help="Need to be at least codes_dim + 1",
        )
        parser.add_argument(
            "--prior_mu_0",
            type=str,
            default="data_mean",
        )
        parser.add_argument(
            "--prior_sigma_choice",
            type=str,
            default="isotropic",
            choices=["isotropic","iso_005", "iso_001", "iso_0001", "data_std","data_std_v2","density_data_std","gt_data_std",'dynamic_data_std'],
        )
        parser.add_argument(
            "--prior_sigma_scale",
            type=float,
            default=".005",
        )
        parser.add_argument(
            "--prior_sigma_scale_step",
            type=float,
            default=1.,
            help="add to change sigma scale between alternations"
        )
        parser.add_argument(
            "--compute_params_every",
            type=int,
            help="How frequently to compute the clustering params (mus, sub, pis)",
            default=1,
        )
        parser.add_argument(
            "--start_computing_params",
            type=int,
            help="When to start to compute the clustering params (mus, sub, pis)",
            default=25,
        )
        parser.add_argument(
            "--cluster_loss",
            type=str,
            help="What kind og loss to use",
            default="KL_GMM_2",
            choices=["diag_NIG", "isotropic", "isotropic_2", "isotropic_3", "isotropic_4", "KL_GMM_2","cosine_dissimilarity","KL_GMM_2_distord_log_mapping"],
        )
        parser.add_argument(
            "--subcluster_loss",
            type=str,
            help="What kind og loss to use",
            default="KL_GMM_2",
            choices=["cosine","diag_NIG", "isotropic", "KL_GMM_2"],
        )
        
        parser.add_argument(
            "--use_priors_for_net_params_init",
            type=bool,
            default=False,
            help="when the net is re-initialized after an AE round, if centers are given, if True it will initialize the covs and the pis using the priors, if false it will compute them using min dist."
        )
        parser.add_argument(
            "--ignore_subclusters",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--log_metrics_at_train",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--evaluate_every_n_epochs",
            type=int,
            default=5,
            help="How often to evaluate the net"
        )
        return parser
