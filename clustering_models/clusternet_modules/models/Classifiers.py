#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import string


class MLP_Classifier(nn.Module):
    def __init__(self, hparams, codes_dim=320, k=None, weights_fc1=None, weights_fc2=None, bias_fc1=None, bias_fc2=None,):
        super(MLP_Classifier, self).__init__()
        if k is None:
            self.k = hparams.init_k
        else:
            self.k = k

        self.codes_dim = codes_dim
        self.hidden_dims = hparams.clusternet_hidden_layer_list
        self.last_dim = self.hidden_dims[-1]
        self.class_fc1 = nn.Linear(self.codes_dim, self.hidden_dims[0])
        hidden_modules = []
        for i in range(len(self.hidden_dims) - 1):
            hidden_modules.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            hidden_modules.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_modules)
        self.class_fc2 = nn.Linear(self.hidden_dims[-1], self.k)
        print(self.hidden_layers)
        self.device = "cuda" if torch.cuda.is_available() and hparams.gpus is not None else "cpu"

        if weights_fc1 is not None:
            self.class_fc1.weight.data = weights_fc1
        if weights_fc2 is not None:
            self.class_fc2.weight.data = weights_fc2
        if bias_fc1 is not None:
            self.class_fc1.bias.data = bias_fc1
        if bias_fc2 is not None:
            self.class_fc2.bias.data = bias_fc2

        self.softmax_norm = hparams.softmax_norm
        # Initialize cluster labels
        uppercase_letters = list(string.ascii_uppercase)
        num_letters = len(uppercase_letters)
        self.cluster_labels = {}
        for i in range(self.k):
            quotient, remainder = divmod(i, num_letters)
            label = uppercase_letters[remainder] * (quotient + 1)
            self.cluster_labels[i] = label

    def _check_nan(self, x, num):
        if torch.isnan(x).any():
            print(f"forward {num}")
            if torch.isnan(self.class_fc1.weight.data).any():
                print("fc1 weights contain nan")
            elif torch.isnan(self.class_fc1.bias.data).any():
                print("fc1 bias contain nan")
            elif torch.isnan(self.class_fc2.weight.data).any():
                print("fc2 weights contain nan")
            elif torch.isnan(self.class_fc2.bias.data).any():
                print("fc2 bias contain nan")
            else:
                print("no weights are nan!")

    def forward(self, x):
        x = x.view(-1, self.codes_dim)
        x = F.relu(self.class_fc1(x))
        x = self.hidden_layers(x)
        x = self.class_fc2(x)
        x = torch.mul(x, self.softmax_norm)
        return F.softmax(x, dim=1)
    

    def update_K_split_Nsubcluster(self, split_decisions, init_new_weights="same", subclusters_nets=None, n_sub_list=None):
        """
        Generalized update_K_split function to handle variable number of subclusters per cluster.
    
        Parameters:
        - split_decisions: List of booleans indicating whether to split each cluster.
        - init_new_weights: Method for initializing new weights ("same", "random").
        - subclusters_nets: Subclustering network (if any).
        - n_sub_list: List containing the number of subclusters per cluster.
        """
        print('INTO CLUSTER UPDATE K SPLIT:')
        class_fc2 = self.class_fc2
        split_decisions_tensor = torch.tensor(split_decisions)
        mus_ind_to_split = torch.nonzero(split_decisions_tensor, as_tuple=False).squeeze()
        if mus_ind_to_split.dim() == 0:
            mus_ind_to_split = mus_ind_to_split.unsqueeze(0)
    
        num_splits = len(mus_ind_to_split)
        total_new_clusters = sum(n_sub_list[k] for k in mus_ind_to_split.tolist())
    
        # Update the number of clusters (self.k)
        self.k = self.k - num_splits + total_new_clusters
    
        with torch.no_grad():
            # Create a new Linear layer with the updated number of clusters
            self.class_fc2 = nn.Linear(self.last_dim, self.k)
    
            # Adjust weights
            not_split_indices = (~split_decisions_tensor).nonzero(as_tuple=False).squeeze()
            split_indices = split_decisions_tensor.nonzero(as_tuple=False).squeeze()
            
            # Ensure split_indices is at least 1D
            if split_indices.dim() == 0:
                split_indices = torch.tensor([split_indices.item()])
            
            # **New Code:** Ensure not_split_indices is at least 1D
            if not_split_indices.dim() == 0:
                not_split_indices = not_split_indices.unsqueeze(0)
    
            weights_not_split = class_fc2.weight.data[not_split_indices, :]
            weights_split = class_fc2.weight.data[split_indices, :]
            print('SPLIT INDICES IN UPDATE K SPLIT CLUSTERNET:', split_indices)
            print('split_decisions_tensor: ', split_decisions_tensor)
            
            # Initialize new weights for the split clusters
            new_weights = self._initialize_weights_split(
                weights_split, split_indices, init_new_weights=init_new_weights, subclusters_nets=subclusters_nets, n_sub_list=n_sub_list
            )
    
            # Concatenate weights for clusters not split and the new weights for split clusters
            self.class_fc2.weight.data = torch.cat(
                [weights_not_split, new_weights], dim=0
            )
    
            # Adjust biases
            biases_not_split = class_fc2.bias.data[not_split_indices]
            biases_split = class_fc2.bias.data[split_indices]
    
            # Initialize new biases for the split clusters
            new_biases = self._initialize_bias_split(
                biases_split, split_indices, init_new_weights=init_new_weights, subclusters_nets=subclusters_nets, n_sub_list=n_sub_list
            )
    
            # Concatenate biases for clusters not split and the new biases for split clusters
            self.class_fc2.bias.data = torch.cat(
                [biases_not_split, new_biases], dim=0
            )
            
            # Update cluster labels
            updated_cluster_labels = {}
            new_idx = 0
            not_split_indices = (~split_decisions_tensor).nonzero(as_tuple=False).squeeze()
            if not_split_indices.dim() == 0:
                not_split_indices = not_split_indices.unsqueeze(0)
            for idx in not_split_indices.tolist():
                updated_cluster_labels[new_idx] = self.cluster_labels[idx]
                new_idx += 1 
            split_indices = split_decisions_tensor.nonzero(as_tuple=False).squeeze()
            if split_indices.dim() == 0:
                split_indices = torch.tensor([split_indices.item()])
            for idx in split_indices.tolist():
                num_new_clusters = n_sub_list[idx]
                old_label = self.cluster_labels[idx]
                for i in range(num_new_clusters):
                    new_label = f"{old_label}{i + 1}"
                    updated_cluster_labels[new_idx] = new_label
                    new_idx += 1
            self.cluster_labels = updated_cluster_labels



    
    def log_cluster_labels(self):
        print("Cluster Labels:")
        for idx, label in self.cluster_labels.items():
            print(f"Cluster MLP index {idx}: Label '{label}'")
    
    
    def update_K_split(self,
                       split_decisions,
                       accepted_subclusters,
                       init_new_weights="same",
                       subclusters_nets=None,
                       n_sub_list=None):
        """
        Update 'class_fc2' so that any cluster k with split_decisions[k] == True
        is replaced by the subclusters in accepted_subclusters[k].

        Parameters
        ----------
        split_decisions : list[bool] or torch.BoolTensor
            Which main clusters are splitting.
        accepted_subclusters : list[list[int]]
            For each main cluster k, a list of subcluster indices that become new main clusters.
            If empty, that cluster is effectively removed (or remains unsplit if split_decisions[k] == False).
        init_new_weights : str
            Weight initialization method for new subclusters ("same", "random", "subclusters", ...).
        subclusters_nets : optional
            A reference to your subclustering net (if you need it for weight init).
        n_sub_list : optional
            Original number of subclusters per cluster (still optional).
        """
        print('INTO CLUSTER UPDATE K SPLIT:')

        # Convert to a torch tensor if needed
        if not isinstance(split_decisions, torch.Tensor):
            split_decisions_tensor = torch.tensor(split_decisions, dtype=torch.bool)
        else:
            split_decisions_tensor = split_decisions.clone()

        # Identify the clusters that are actually splitting
        mus_ind_to_split = torch.nonzero(split_decisions_tensor, as_tuple=False).squeeze()
        if mus_ind_to_split.dim() == 0:
            mus_ind_to_split = mus_ind_to_split.unsqueeze(0)

        # Count how many new clusters we will add
        # i.e., sum up the lengths of accepted_subclusters[k] for each k that splits
        total_new_clusters = 0
        for k_idx in mus_ind_to_split.tolist():
            total_new_clusters += len(accepted_subclusters[k_idx])

        # Each cluster that splits is removed from the classifier, 
        # then we add as many new clusters as accepted_subclusters.
        num_splits = len(mus_ind_to_split)
        old_k = self.k
        self.k = self.k - num_splits + total_new_clusters
        print(f"update_K_split: old K={old_k}, new K={self.k}")

        # Save old weights/bias for reusing or referencing
        old_class_fc2 = self.class_fc2
        old_weights = old_class_fc2.weight.data.clone()
        old_biases = old_class_fc2.bias.data.clone()

        # Create a new Linear layer with the updated number of clusters
        self.class_fc2 = nn.Linear(self.last_dim, self.k).to(self.device)

        # Indices of clusters that are not splitting
        not_split_indices = (~split_decisions_tensor).nonzero(as_tuple=False).squeeze()
        if not_split_indices.dim() == 0:
            not_split_indices = not_split_indices.unsqueeze(0)

        # Indices of clusters that are splitting
        split_indices = (split_decisions_tensor).nonzero(as_tuple=False).squeeze()
        if split_indices.dim() == 0:
            split_indices = split_indices.unsqueeze(0)

        # Gather old weights/bias for the clusters that are NOT splitting
        weights_not_split = old_weights[not_split_indices, :]  # shape [#not_split, last_dim]
        biases_not_split = old_biases[not_split_indices]       # shape [#not_split]

        # Build new weights/bias for subclusters from splitting clusters
        new_weights_list = []
        new_biases_list = []

        # We'll iterate in parallel over split_indices and accepted_subclusters
        for cluster_idx in split_indices.tolist():
            # The old row is old_weights[cluster_idx, :], but that cluster is being replaced
            old_weight_cluster = old_weights[cluster_idx, :]
            old_bias_cluster = old_biases[cluster_idx]

            # Which subclusters are accepted for this main cluster?
            sub_idx_list = accepted_subclusters[cluster_idx]
            if len(sub_idx_list) == 0:
                # If none are accepted, effectively remove that cluster entirely
                continue

            # For each accepted subcluster, create a row of weights/bias
            # using the chosen initialization method
            for sub_idx in sub_idx_list:
                # EXAMPLE: copy from the old cluster ("same") or random init
                if init_new_weights == "same":
                    w_new = old_weight_cluster.clone().unsqueeze(0)   # shape [1, last_dim]
                    b_new = old_bias_cluster.clone().unsqueeze(0)     # shape [1]
                elif init_new_weights == "random":
                    w_new = torch.randn(1, old_weight_cluster.shape[0], device=self.device)
                    b_new = torch.randn(1, device=self.device)
                elif init_new_weights == "subclusters" and subclusters_nets is not None:
                    # Optionally pull from some specialized subcluster network
                    w_new, b_new = self._extract_subcluster_weights(subclusters_nets, cluster_idx, sub_idx)
                    # Should return shape [1, last_dim] for w_new, shape [1] for b_new
                else:
                    raise NotImplementedError("Unknown init_new_weights method or missing subclusters_nets.")

                new_weights_list.append(w_new)
                new_biases_list.append(b_new)

        # If we have new subcluster rows, concatenate them
        if len(new_weights_list) > 0:
            cat_new_weights = torch.cat(new_weights_list, dim=0)
            cat_new_biases = torch.cat(new_biases_list, dim=0)
        else:
            cat_new_weights = torch.empty((0, old_weights.size(1)), device=self.device)
            cat_new_biases = torch.empty((0,), device=self.device)

        # Final weights/bias: keep the not-split clusters + newly added subclusters
        final_weights = torch.cat([weights_not_split, cat_new_weights], dim=0)
        final_biases  = torch.cat([biases_not_split,  cat_new_biases],  dim=0)

        # Assign them to our new layer
        self.class_fc2.weight.data = final_weights
        self.class_fc2.bias.data   = final_biases

        # ----------------------------
        # Update cluster labels
        # ----------------------------
        updated_cluster_labels = {}
        new_idx = 0

        # 1) Keep labels for clusters that didn't split
        for old_idx in not_split_indices.tolist():
            updated_cluster_labels[new_idx] = self.cluster_labels[old_idx]
            new_idx += 1

        # 2) For each splitting cluster, add subcluster labels
        for old_idx in split_indices.tolist():
            sub_idx_list = accepted_subclusters[old_idx]
            old_label = self.cluster_labels[old_idx]
            # If no subclusters accepted, we skip it entirely
            for i, sub_idx in enumerate(sub_idx_list):
                # Example: label them oldLabel1, oldLabel2, ...
                new_label = f"{old_label}{i + 1}"
                updated_cluster_labels[new_idx] = new_label
                new_idx += 1

        self.cluster_labels = updated_cluster_labels

        print(f"Final new shape for class_fc2: {self.class_fc2.weight.shape}")
        print(f"Updated cluster labels: {self.cluster_labels}")

    # ----------------------------------------------------
    # Example helper for subcluster-based weight extraction
    # Only needed if you do "init_new_weights='subclusters'".
    # ----------------------------------------------------
    def _extract_subcluster_weights(self, subclusters_nets, cluster_idx, sub_idx):
        """
        Example function to show how you'd pull weights for subcluster (cluster_idx, sub_idx)
        from a 'subclusters_net'. The shapes returned should be [1, last_dim] for weights
        and [1] for biases, so we can cat them into class_fc2 easily.
        """
        # This is just a placeholder. Fill in your own logic.
        # For instance, you might do something like:
        #  w = subclusters_nets[cluster_idx].class_fc2.weight[sub_idx, :].clone()
        #  b = subclusters_nets[cluster_idx].class_fc2.bias[sub_idx].clone()
        # Make sure to reshape to [1, ...].
        raise NotImplementedError("Implement logic for extracting subcluster weights here.")
         
    def update_K_merge(self, inds_to_mask, mus_lists_to_merge, highest_ll_mus, clusters_to_suppress, init_new_weights="same"):
        """
        Updates the clustering network after clusters have been merged and suppressed.
    
        Args:
            inds_to_mask (torch.Tensor): Boolean tensor indicating clusters to be merged or suppressed.
            mus_lists_to_merge (list): List of lists containing indices of clusters to merge.
            highest_ll_mus (list): Indices of clusters with the highest log-likelihood in each merge group.
            clusters_to_suppress (list): Indices of clusters to suppress (remove).
            init_new_weights (str): Method to initialize new weights ("same", "random", "average").
        """
        print("DEBUG: Updating clustering network after merge")
    
        # Combine inds_to_mask and clusters_to_suppress
        total_inds_to_mask = inds_to_mask.clone()
        for idx in clusters_to_suppress:
            total_inds_to_mask[idx] = True
    
        # Clusters not merged or suppressed
        indices_not_merged = (~total_inds_to_mask).nonzero(as_tuple=False).squeeze().tolist()
        if isinstance(indices_not_merged, int):
            indices_not_merged = [indices_not_merged]
    
        # Update k
        num_clusters_merged = sum(len(group) for group in mus_lists_to_merge)
        num_new_clusters = len(mus_lists_to_merge)
        num_clusters_suppressed = len(clusters_to_suppress)
        self.k = self.k - num_clusters_merged + num_new_clusters - num_clusters_suppressed
    
        with torch.no_grad():
            # Create a new Linear layer with the updated number of clusters
            new_class_fc2 = nn.Linear(self.last_dim, self.k).to(device=self.device)
    
            # Adjust weights
            weights_not_merged = self.class_fc2.weight.data[indices_not_merged, :]
            weights_merged = []
            for merge_group, highest_ll_idx in zip(mus_lists_to_merge, highest_ll_mus):
                if init_new_weights == "same":
                    # Use weights of the cluster with highest log-likelihood
                    weights_merged.append(self.class_fc2.weight.data[highest_ll_idx, :].unsqueeze(0))
                elif init_new_weights == "average":
                    # Average weights of merged clusters
                    weights = [self.class_fc2.weight.data[idx, :] for idx in merge_group]
                    weights_avg = torch.mean(torch.stack(weights), dim=0).unsqueeze(0)
                    weights_merged.append(weights_avg)
                elif init_new_weights == "random":
                    # Initialize weights randomly
                    weights_rand = torch.randn(1, self.class_fc2.weight.data.shape[1], device=self.device)
                    weights_merged.append(weights_rand)
                else:
                    raise NotImplementedError("Unknown initialization method.")
    
            # Concatenate weights
            if weights_merged:
                weights_merged = torch.cat(weights_merged, dim=0)
                new_class_fc2.weight.data = torch.cat([weights_not_merged, weights_merged], dim=0)
            else:
                new_class_fc2.weight.data = weights_not_merged
    
            # Adjust biases
            biases_not_merged = self.class_fc2.bias.data[indices_not_merged]
            biases_merged = []
            for merge_group, highest_ll_idx in zip(mus_lists_to_merge, highest_ll_mus):
                if init_new_weights == "same":
                    biases_merged.append(self.class_fc2.bias.data[highest_ll_idx].unsqueeze(0))
                elif init_new_weights == "average":
                    biases = [self.class_fc2.bias.data[idx] for idx in merge_group]
                    biases_avg = torch.mean(torch.stack(biases)).unsqueeze(0)
                    biases_merged.append(biases_avg)
                elif init_new_weights == "random":
                    biases_rand = torch.zeros(1, device=self.device)
                    biases_merged.append(biases_rand)
                else:
                    raise NotImplementedError("Unknown initialization method.")
    
            # Concatenate biases
            if biases_merged:
                biases_merged = torch.cat(biases_merged, dim=0)
                new_class_fc2.bias.data = torch.cat([biases_not_merged, biases_merged], dim=0)
            else:
                new_class_fc2.bias.data = biases_not_merged
    
            # Assign the new class_fc2
            self.class_fc2 = new_class_fc2
    
        # Update cluster labels
        updated_cluster_labels = {}
        new_idx = 0
        for idx in indices_not_merged:
            updated_cluster_labels[new_idx] = self.cluster_labels[idx]
            new_idx += 1
        for merge_group, highest_ll_idx in zip(mus_lists_to_merge, highest_ll_mus):
            # Assign a new label to the merged cluster
            merged_label = ''.join([self.cluster_labels[idx] for idx in merge_group])
            updated_cluster_labels[new_idx] = merged_label
            new_idx += 1
    
        self.cluster_labels = updated_cluster_labels
    
    
    def _initialize_weights_split(self, weights_split, split_indices, init_new_weights, subclusters_nets=None, n_sub_list=None):
        new_weights = []
        idx_split = 0
        for idx in split_indices.tolist():
            num_new_clusters = n_sub_list[idx]
            weight_k = weights_split[idx_split]  # Get the weight of the cluster being split
            idx_split += 1
    
            if init_new_weights == "same":
                # Duplicate the weight for the new clusters
                weights_new_k = weight_k.unsqueeze(0).repeat(num_new_clusters, 1)
            elif init_new_weights == "random":
                # Initialize new weights randomly
                weights_new_k = torch.FloatTensor(num_new_clusters, weight_k.shape[0]).uniform_(-1., 1.).to(device=self.device)
            elif init_new_weights == "subclusters" and subclusters_nets is not None:
                # Initialize weights from subclustering network
                weights_new_k = self._extract_subcluster_weights(
                    subclusters_nets, idx, num_new_clusters
                )
            else:
                raise NotImplementedError("Unknown initialization method or missing subclusters_nets.")
            new_weights.append(weights_new_k)
        return torch.cat(new_weights, dim=0)


    def _initalize_weights_merge(self, weights_merged, merge_decisions, highest_ll, init_new_weight="same"):
        if init_new_weight == "same":
            # Take the weights of the cluster with highest likelihood
            ll = [i[0].item() for i in highest_ll]
            return self.class_fc2.weight.data[ll, :]
        elif init_new_weight == "random":
            return torch.FloatTensor(len(highest_ll), weights_merged.shape[1]).uniform_(-1., 1).to(device=self.device)
        elif init_new_weight == "average":
            raise NotImplementedError()
        else:
            raise NotImplementedError
    def _initialize_bias_split(self, biases_split, split_indices, init_new_weights, subclusters_nets=None, n_sub_list=None):
        new_biases = []
        idx_split = 0
        for idx in split_indices.tolist():
            num_new_clusters = n_sub_list[idx]
            bias_k = biases_split[idx_split]  # Get the bias of the cluster being split
            idx_split += 1
    
            if init_new_weights == "same":
                # Duplicate the bias for the new clusters
                biases_new_k = bias_k.repeat(num_new_clusters)
            elif init_new_weights == "random":
                # Initialize new biases to zeros
                biases_new_k = torch.zeros(num_new_clusters).to(device=self.device)
            elif init_new_weights == "subclusters" and subclusters_nets is not None:
                # Initialize biases from subclustering network
                biases_new_k = self._extract_subcluster_biases(
                    subclusters_nets, idx, num_new_clusters
                )
            else:
                raise NotImplementedError("Unknown initialization method or missing subclusters_nets.")
            new_biases.append(biases_new_k)
        return torch.cat(new_biases, dim=0)
    def _extract_subcluster_weights(self, subclusters_nets, cluster_idx, num_subclusters):
        """
        Extracts the weights for the subclusters of a given cluster from the subclustering network.
    
        Parameters:
        - subclusters_nets: The subclustering network.
        - cluster_idx: The index of the cluster being split.
        - num_subclusters: The number of subclusters for this cluster.
    
        Returns:
        - weights_new_k: Tensor containing the weights for the new clusters.
        """
        hidden_dim = subclusters_nets.hidden_dim
        subcluster_offset = sum(subclusters_nets.subclusters_per_cluster[:cluster_idx])
        subcluster_indices = list(range(subcluster_offset, subcluster_offset + num_subclusters))
        hidden_unit_indices = list(range(hidden_dim * cluster_idx, hidden_dim * (cluster_idx + 1)))

        # Extract weights and reshape to match the classifier's expected dimensions
        weights_k = subclusters_nets.class_fc2.weight.data[subcluster_indices][:, hidden_unit_indices].clone()
        return weights_k  # Shape: [num_subclusters, hidden_dim]

    def _extract_subcluster_biases(self, subclusters_nets, cluster_idx, num_subclusters):
        """
        Extracts the biases for the subclusters of a given cluster from the subclustering network.
    
        Parameters:
        - subclusters_nets: The subclustering network.
        - cluster_idx: The index of the cluster being split.
        - num_subclusters: The number of subclusters for this cluster.
    
        Returns:
        - biases_new_k: Tensor containing the biases for the new clusters.
        """
        subcluster_offset = sum(subclusters_nets.subclusters_per_cluster[:cluster_idx])
        subcluster_indices = list(range(subcluster_offset, subcluster_offset + num_subclusters))
    
        biases_k = subclusters_nets.class_fc2.bias.data[subcluster_indices].clone()
        return biases_k  # Shape: [num_subclusters]


    def _initalize_bias_merge(self, bias_merged, merge_decisions, highest_ll, init_new_weight="same"):
        if init_new_weight == "same":
            # take the biases of the highest likelihood
            ll = [i[0].item() for i in highest_ll]
            return self.class_fc2.bias.data[ll]
        elif init_new_weight == "random":
            return torch.zeros(len(highest_ll)).to(device=self.device)
        elif init_new_weight == "average":
            raise NotImplementedError
        else:
            raise NotImplementedError


class Subclustering_net_duplicating(nn.Module):
    def __init__(self, hparams, codes_dim=320, k=None, subclusters_per_cluster=None):
        super(Subclustering_net, self).__init__()
        
        #print("DEBUG: Initializing Subclustering_net")
        #print(f"DEBUG: hparams = {hparams}")
        #print(f"DEBUG: codes_dim = {codes_dim}")
        #print(f"DEBUG: k = {k}")
        #print(f"DEBUG: subclusters_per_cluster = {subclusters_per_cluster}")

        # Use the provided number of main clusters or fallback to initial K
        if k is None:
            self.K = hparams.init_k
            print(f"DEBUG: k is None, using hparams.init_k = {self.K}")
        else:
            self.K = k
            print(f"DEBUG: Using provided k = {self.K}")

        # If no subclusters are provided, we use 2 per cluster as the default
        if subclusters_per_cluster is None:
            subclusters_per_cluster = [2] * self.K  # Default to 2 subclusters per cluster
            print(f"DEBUG: subclusters_per_cluster not provided, defaulting to {subclusters_per_cluster}")
        else:
            print(f"DEBUG: Using provided subclusters_per_cluster = {subclusters_per_cluster}")

        # Total number of subclusters is the sum of the list
        self.subclusters_per_cluster = subclusters_per_cluster
        self.total_subclusters = sum(subclusters_per_cluster)
        print(f"DEBUG: total_subclusters = {self.total_subclusters}")

        self.codes_dim = codes_dim
        self.hparams = hparams
        self.hidden_dim = 50
        self.softmax_norm = self.hparams.softmax_norm
        self.device = "cuda" if torch.cuda.is_available() and hparams.gpus is not None else "cpu"
        print(f"DEBUG: Using device = {self.device}")

        # the subclustering net will be a stacked version of the clustering net
        self.class_fc1 = nn.Linear(self.codes_dim, self.hidden_dim * self.K).to(self.device)
        self.class_fc2 = nn.Linear(self.hidden_dim * self.K, self.total_subclusters).to(self.device)
        #print(f"DEBUG: Initialized class_fc1 with shape {self.class_fc1.weight.shape}")
        #print(f"DEBUG: Initialized class_fc2 with shape {self.class_fc2.weight.shape}")

        # Create a gradient mask for detaching different subclustering nets
        gradient_mask_fc2 = torch.zeros(self.hidden_dim * self.K, self.total_subclusters).to(self.device)
        print(f"DEBUG: Created initial gradient_mask_fc2 with shape {gradient_mask_fc2.shape}")

        subcluster_offset = 0  # To handle different numbers of subclusters per cluster
        for k_idx in range(self.K):
            num_subclusters = subclusters_per_cluster[k_idx]
            gradient_mask_fc2[self.hidden_dim * k_idx:self.hidden_dim * (k_idx + 1), subcluster_offset: subcluster_offset + num_subclusters] = 1
            print(f"DEBUG: Setting gradient mask for cluster {k_idx} with {num_subclusters} subclusters")
            subcluster_offset += num_subclusters

        #print(f"DEBUG: Final gradient_mask_fc2 =\n{gradient_mask_fc2}")
        self.class_fc2.weight.data *= gradient_mask_fc2.T
        #print(f"DEBUG: After applying gradient mask, class_fc2.weight.data =\n{self.class_fc2.weight.data}")

        # Register gradient hook
        def grad_hook(grad):
            #print("DEBUG: Gradient hook triggered for class_fc2.weight")
            #print(f"DEBUG: Original grad shape = {grad.shape}")
            modified_grad = grad.mul_(gradient_mask_fc2.T)
            #print(f"DEBUG: Modified grad shape = {modified_grad.shape}")
            return modified_grad

        self.class_fc2.weight.register_hook(grad_hook)

    def forward(self, X, hard_assign):
        X = self.reshape_input(X, hard_assign)
        X = F.relu(self.class_fc1(X))
        X = self.class_fc2(X)
        X = torch.mul(X, self.softmax_norm)
        return F.softmax(X, dim=1)

    def reshape_input(self, X, hard_assign):
        # each input (batch_size X codes_dim) will be padded with zeros to insert to the stacked subnets
        X = X.view(-1, self.codes_dim)
        new_batch = torch.zeros(X.size(0), self.K, X.size(1))
        for k in range(self.K):
            new_batch[hard_assign == k, k, :] = X[hard_assign == k]
        new_batch = new_batch.view(X.size(0), -1)  # in s_batch X d * K
        return new_batch


class Subclustering_net(nn.Module):
    # Generalized for a variable number of subclusters
    # SHAPE is input dim -> 50 * K -> sum(n_sub)
    def __init__(self, hparams, codes_dim=320, k=None, subclusters_per_cluster=None):
        super(Subclustering_net, self).__init__()
        

        # Use the provided number of main clusters or fallback to initial K
        if k is None:
            self.K = hparams.init_k
        else:
            self.K = k

        # If no subclusters are provided, we use 2 per cluster as the default
        if subclusters_per_cluster is None:
            subclusters_per_cluster = [2] * self.K  # Default to 2 subclusters per cluster
        else:
            print(f"DEBUG: Using provided subclusters_per_cluster = {subclusters_per_cluster}")

        # Total number of subclusters is the sum of the list
        self.subclusters_per_cluster = subclusters_per_cluster
        self.total_subclusters = sum(subclusters_per_cluster)
        print(f"DEBUG: total_subclusters = {self.total_subclusters}")

        self.codes_dim = codes_dim
        self.hparams = hparams
        self.hidden_dim = 50
        self.softmax_norm = self.hparams.softmax_norm
        self.device = "cuda" if torch.cuda.is_available() and hparams.gpus is not None else "cpu"
        print(f"DEBUG: Using device = {self.device}")

        # the subclustering net will be a stacked version of the clustering net
        self.class_fc1 = nn.Linear(self.codes_dim, self.hidden_dim * self.K).to(self.device)
        self.class_fc2 = nn.Linear(self.hidden_dim * self.K, self.total_subclusters).to(self.device)
        print(f"DEBUG: Initialize class_fc1 with shape {self.class_fc1.weight.shape}")
        print(f"DEBUG: Initialize class_fc2 with shape {self.class_fc2.weight.shape}")
        # Create a gradient mask for detaching different subclustering nets
        gradient_mask_fc2 = torch.zeros(self.hidden_dim * self.K, self.total_subclusters).to(self.device)
        print(f"DEBUG: Created initial gradient_mask_fc2 with shape {gradient_mask_fc2.shape}")
        subcluster_offset = 0  # To handle different numbers of subclusters per cluster
        for k_idx in range(self.K):
            num_subclusters = subclusters_per_cluster[k_idx]
            gradient_mask_fc2[self.hidden_dim * k_idx:self.hidden_dim * (k_idx + 1), subcluster_offset: subcluster_offset + num_subclusters] = 1
            print(f"DEBUG: Setting gradient mask for cluster {k_idx} with {num_subclusters} subclusters")
            subcluster_offset += num_subclusters

        print(f"DEBUG: Final gradient_mask_fc2 =\n{gradient_mask_fc2}")
        self.class_fc2.weight.data *= gradient_mask_fc2.T
        print(f"DEBUG: After applying gradient mask, class_fc2.weight.data =\n{self.class_fc2.weight.data}")
        self.cluster_labels = {}
        self.subcluster_labels = {}
        uppercase_letters = list(string.ascii_uppercase)
        num_letters = len(uppercase_letters)
        for i in range(self.K):
            quotient, remainder = divmod(i, num_letters)
            label = uppercase_letters[remainder] * (quotient + 1)
            self.cluster_labels[i] = label
            self.subcluster_labels[i] = {j: label + str(j + 1) for j in range(self.subclusters_per_cluster[i])}

        # weights are zero and their grad will always be 0 so won't change
    
    def forward(self, X):
        # A) Re-check that input is finite (double-safe)
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise RuntimeError("ERROR: Forward input `X` to class_fc1 is NaN/Inf")
    
        # B) First Linear Layer (class_fc1)
        X1 = self.class_fc1(X)
        if torch.isnan(X1).any() or torch.isinf(X1).any():
            # Log summary statistics to understand magnitude
            print(f"? class_fc1 output is invalid (NaN/Inf). stats: min={X1.min().item()}, max={X1.max().item()}, mean={X1.mean().item()}")
            raise RuntimeError("ERROR: class_fc1 produced NaN/Inf")
    
        # C) ReLU Nonlinearity (cannot introduce NaN on finite input)
        X2 = F.relu(X1)
    
        # D) Second Linear Layer (class_fc2)
        X3 = self.class_fc2(X2)
        if torch.isnan(X3).any() or torch.isinf(X3).any():
            # Again, log stats to see approximate magnitude
            print(f"? class_fc2 output is invalid (NaN/Inf). stats: min={X3.min().item()}, max={X3.max().item()}, mean={X3.mean().item()}")
            raise RuntimeError("ERROR: class_fc2 produced NaN/Inf")
    
        return X3

    def forward_adecoch(self, X):
        # Attach hooks to monitor gradients dynamically
        """
        def attach_gradient_hooks(layer, name):
            if not hasattr(layer.weight, 'hook_installed'):
                def grad_hook(grad):
                    # Check for NaN gradients
                    if torch.isnan(grad).any():
                        print(f"NaN detected in gradients for {name}.weight!")
                    
                    # Check for Inf gradients
                    if torch.isinf(grad).any():
                        print(f"Inf detected in gradients for {name}.weight!")
                    
                    # Check for abnormally large gradients
                    large_grad_threshold = 1e3  # Adjust based on expected gradient scale
                    large_grad_indices = (grad.abs() > large_grad_threshold).nonzero(as_tuple=False)
                    if large_grad_indices.numel() > 0:
                        print(f"Large gradients detected in {name}.weight at indices: {large_grad_indices.tolist()}")
    
                    # Log gradient statistics
                    print(f"{name} Gradient stats - min: {grad.min().item()}, max: {grad.max().item()}, mean: {grad.mean().item()}")
    
                # Attach the hook
                layer.weight.register_hook(grad_hook)
                layer.weight.hook_installed = True
    
                # Optional: Attach the same hook to biases
                if layer.bias is not None:
                    layer.bias.register_hook(grad_hook)
    
        # Attach hooks to class_fc1 and class_fc2
        attach_gradient_hooks(self.class_fc1, "class_fc1")
        attach_gradient_hooks(self.class_fc2, "class_fc2")
        """
        # Forward pass through the network
        X = F.relu(self.class_fc1(X))
        X = self.class_fc2(X)
      
        return X

    
    def _initalize_weights_split(self, weights_old, init_new_weights, num_new_subclusters):
        print("DEBUG: Initializing weights after split")
        print(f"DEBUG: weights_old shape = {weights_old.shape}")
        print(f"DEBUG: init_new_weights = {init_new_weights}")
        print(f"DEBUG: num_new_subclusters = {num_new_subclusters}")

        if init_new_weights == "same":
            # Duplicate the old weights to match the number of new subclusters
            old_num_subclusters = weights_old.shape[0]
            print(f"DEBUG: old_num_subclusters = {old_num_subclusters}")
            if old_num_subclusters == num_new_subclusters:
                # Same number of subclusters, just copy
                new_weights = weights_old.clone()
                print("DEBUG: Same number of subclusters, cloned weights")
            else:
                repeats = num_new_subclusters // old_num_subclusters
                remainder = num_new_subclusters % old_num_subclusters
                print(f"DEBUG: repeats = {repeats}, remainder = {remainder}")
                new_weights = weights_old.repeat(repeats, 1)
                if remainder > 0:
                    new_weights = torch.cat([new_weights, weights_old[:remainder]], dim=0)
                    print(f"DEBUG: Concatenated {remainder} additional weights")
        elif init_new_weights == "random":
            # Initialize randomly
            #new_weights = torch.randn(num_new_subclusters, weights_old.shape[1], device=self.device)
            new_weights = torch.FloatTensor(num_new_subclusters, weights_old.shape[1]).uniform_(-1., 1).to(device=self.device)
            print(f"DEBUG: Randomly initialized new_weights with shape {new_weights.shape}")
        else:
            raise NotImplementedError("Unknown init_new_weights method: {}".format(init_new_weights))
        
        print(f"DEBUG: new_weights shape = {new_weights.shape}")
        return new_weights
    
    def _initalize_bias_split(self, biases_old, init_new_weights, num_new_subclusters):
        print("DEBUG: Initializing biases after split")
        print(f"DEBUG: biases_old shape = {biases_old.shape}")
        print(f"DEBUG: init_new_weights = {init_new_weights}")
        print(f"DEBUG: num_new_subclusters = {num_new_subclusters}")

        if init_new_weights == "same":
            # Duplicate the old biases to match the number of new subclusters
            old_num_subclusters = biases_old.shape[0]
            print(f"DEBUG: old_num_subclusters = {old_num_subclusters}")
            if old_num_subclusters == num_new_subclusters:
                # Same number of subclusters, just copy
                new_biases = biases_old.clone()
                print("DEBUG: Same number of subclusters, cloned biases")
            else:
                repeats = num_new_subclusters // old_num_subclusters
                remainder = num_new_subclusters % old_num_subclusters
                print(f"DEBUG: repeats = {repeats}, remainder = {remainder}")
                new_biases = biases_old.repeat(repeats)
                if remainder > 0:
                    new_biases = torch.cat([new_biases, biases_old[:remainder]], dim=0)
                    print(f"DEBUG: Concatenated {remainder} additional biases")
        elif init_new_weights == "random":
            # Initialize biases to zero
            new_biases = torch.zeros(num_new_subclusters, device=self.device)
            print(f"DEBUG: Initialized new_biases to zeros with shape {new_biases.shape}")
        else:
            raise NotImplementedError("Unknown init_new_weights method: {}".format(init_new_weights))
        
        print(f"DEBUG: new_biases shape = {new_biases.shape}")
        return new_biases


    
    def update_K_merge(self, inds_to_mask, mus_lists_to_merge, highest_ll_mus, clusters_to_suppress, init_new_weights="highest_ll", n_sub_list_new=None):
        """
        Updates the clustering network after clusters have been merged and suppressed.
    
        Args:
            inds_to_mask (torch.Tensor): Boolean tensor indicating clusters to be merged or suppressed.
            mus_lists_to_merge (list): List of lists containing indices of clusters to merge.
            highest_ll_mus (list): Indices of clusters with the highest log-likelihood in each merge group.
            clusters_to_suppress (list): Indices of clusters to suppress (remove).
            init_new_weights (str): Method to initialize new weights ("highest_ll", "random").
            n_sub_list_new: List containing the new number of subclusters per cluster.
        """
        print("DEBUG: Updating subclustering network after merge")
    
        # Combine inds_to_mask and clusters_to_suppress
        total_inds_to_mask = inds_to_mask.clone()
        for idx in clusters_to_suppress:
            total_inds_to_mask[idx] = True
    
        # Clusters not merged or suppressed
        indices_not_merged = (~total_inds_to_mask).nonzero(as_tuple=False).squeeze().tolist()
        if isinstance(indices_not_merged, int):
            indices_not_merged = [indices_not_merged]
        print(f"DEBUG: Indices not merged or suppressed: {indices_not_merged}")
    
        # Capture the old subclusters before updating
        old_subclusters_per_cluster = self.subclusters_per_cluster.copy()
        print(f"DEBUG: Old subclusters_per_cluster: {old_subclusters_per_cluster}")
    
        # Update subclusters_per_cluster and total_subclusters before resizing layers
        if n_sub_list_new is not None:
            expected_clusters = len(indices_not_merged) + len(mus_lists_to_merge)
            if len(n_sub_list_new) != expected_clusters:
                print(f"ERROR: Length of n_sub_list_new ({len(n_sub_list_new)}) does not match the expected number of clusters ({expected_clusters}).")
                return  # Early exit due to error
            else:
                print(f"DEBUG: Updating subclusters_per_cluster to: {n_sub_list_new}")
                self.subclusters_per_cluster = n_sub_list_new
                self.total_subclusters = sum(self.subclusters_per_cluster)
        else:
            print("ERROR: n_sub_list_new is None. Cannot update subclusters_per_cluster.")
            return  # Early exit due to error
    
        # Update K
        self.K = len(self.subclusters_per_cluster)
        print(f"DEBUG: Updated K to: {self.K}")
    
        with torch.no_grad():
            # Save old weights and biases
            old_class_fc1_weight = self.class_fc1.weight.data.clone()
            old_class_fc1_bias = self.class_fc1.bias.data.clone()
            old_class_fc2_weight = self.class_fc2.weight.data.clone()
            old_class_fc2_bias = self.class_fc2.bias.data.clone()
    
            # Debug weights and biases
            print(f"DEBUG: old_class_fc1_weight shape: {old_class_fc1_weight.shape}")
            print(f"DEBUG: old_class_fc2_weight shape: {old_class_fc2_weight.shape}")
    
            # Update class_fc1 and class_fc2 with new dimensions
            self.class_fc1 = nn.Linear(self.codes_dim, self.hidden_dim * self.K).to(self.device)
            self.class_fc2 = nn.Linear(self.hidden_dim * self.K, self.total_subclusters).to(self.device)
    
            # Initialize new weight and bias tensors
            new_class_fc1_weight = torch.zeros_like(self.class_fc1.weight.data)
            new_class_fc1_bias = torch.zeros_like(self.class_fc1.bias.data)
            new_class_fc2_weight = torch.zeros_like(self.class_fc2.weight.data)
            new_class_fc2_bias = torch.zeros_like(self.class_fc2.bias.data)
    
            # Debug initial weights and biases
            print(f"DEBUG: Initialized new_class_fc1_weight shape: {new_class_fc1_weight.shape}")
            print(f"DEBUG: Initialized new_class_fc2_weight shape: {new_class_fc2_weight.shape}")
    
            # Offset trackers
            new_cluster_idx = 0
            new_subcluster_offset = 0
    
            # Map from old cluster indices to new cluster indices
            cluster_idx_mapping = {}
    
            # Process non-merged and non-suppressed clusters
            for idx in indices_not_merged:
                if not (0 <= idx < len(old_subclusters_per_cluster)):
                    print(f"ERROR: idx {idx} is out of range for old_subclusters_per_cluster with length {len(old_subclusters_per_cluster)}.")
                    continue  # Skip invalid index
    
                num_subclusters_old = old_subclusters_per_cluster[idx]
                num_subclusters_new = self.subclusters_per_cluster[new_cluster_idx]
                print(f"DEBUG: Processing cluster {idx} (new index {new_cluster_idx}) with {num_subclusters_old} old subclusters and {num_subclusters_new} new subclusters.")
    
                old_fc1_start = idx * self.hidden_dim
                old_fc1_end = old_fc1_start + self.hidden_dim
                new_fc1_start = new_cluster_idx * self.hidden_dim
                new_fc1_end = new_fc1_start + self.hidden_dim
    
                # Copy class_fc1 weights and biases
                new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = old_class_fc1_weight[old_fc1_start:old_fc1_end, :]
                new_class_fc1_bias[new_fc1_start:new_fc1_end] = old_class_fc1_bias[old_fc1_start:old_fc1_end]
    
                # Copy class_fc2 weights and biases
                old_fc2_start = sum(old_subclusters_per_cluster[:idx])
                old_fc2_end = old_fc2_start + min(num_subclusters_old, num_subclusters_new)
                new_fc2_start = new_subcluster_offset
                new_fc2_end_existing = new_fc2_start + min(num_subclusters_old, num_subclusters_new)
    
                new_class_fc2_weight[new_fc2_start:new_fc2_end_existing, new_fc1_start:new_fc1_end] = \
                    old_class_fc2_weight[old_fc2_start:old_fc2_end, old_fc1_start:old_fc1_end]
                new_class_fc2_bias[new_fc2_start:new_fc2_end_existing] = old_class_fc2_bias[old_fc2_start:old_fc2_end]
    
                # Initialize additional subclusters if needed
                num_subclusters_to_initialize = num_subclusters_new - min(num_subclusters_old, num_subclusters_new)
                if num_subclusters_to_initialize > 0:
                    new_fc2_end = new_fc2_start + num_subclusters_new
                    new_class_fc2_weight[new_fc2_end_existing:new_fc2_end, new_fc1_start:new_fc1_end] = \
                        torch.randn(num_subclusters_to_initialize, self.hidden_dim, device=self.device)
                    new_class_fc2_bias[new_fc2_end_existing:new_fc2_end] = torch.zeros(num_subclusters_to_initialize, device=self.device)
                else:
                    new_fc2_end = new_fc2_end_existing
    
                # Update mappings and offsets
                cluster_idx_mapping[idx] = new_cluster_idx
                print(f"DEBUG: Mapped old cluster {idx} to new cluster {new_cluster_idx}.")
                new_cluster_idx += 1
                new_subcluster_offset += num_subclusters_new
    
            # Process merged clusters
            print(f"DEBUG: mus_lists_to_merge = {mus_lists_to_merge}")
            print(f"DEBUG: highest_ll_mus = {highest_ll_mus}")
            for merge_group, highest_ll_idx in zip(mus_lists_to_merge, highest_ll_mus):
                if not (0 <= highest_ll_idx < len(old_subclusters_per_cluster)):
                    print(f"ERROR: Invalid highest_ll_idx: {highest_ll_idx}, out of range for old_subclusters_per_cluster with length {len(old_subclusters_per_cluster)}")
                    continue  # Skip this merge group
    
                num_subclusters_old = old_subclusters_per_cluster[highest_ll_idx]
                num_subclusters_new = self.subclusters_per_cluster[new_cluster_idx]
                print(f"DEBUG: Processing merged group {merge_group} (new index {new_cluster_idx}), highest_ll_idx = {highest_ll_idx}, with {num_subclusters_old} old subclusters and {num_subclusters_new} new subclusters.")
    
                old_fc1_start = highest_ll_idx * self.hidden_dim
                old_fc1_end = old_fc1_start + self.hidden_dim
                new_fc1_start = new_cluster_idx * self.hidden_dim
                new_fc1_end = new_fc1_start + self.hidden_dim
    
                if init_new_weights == "highest_ll":
                    # Copy class_fc1 weights and biases
                    new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = old_class_fc1_weight[old_fc1_start:old_fc1_end, :]
                    new_class_fc1_bias[new_fc1_start:new_fc1_end] = old_class_fc1_bias[old_fc1_start:old_fc1_end]
    
                    # Copy class_fc2 weights and biases
                    old_fc2_start = sum(old_subclusters_per_cluster[:highest_ll_idx])
                    old_fc2_end = old_fc2_start + min(num_subclusters_old, num_subclusters_new)
                    new_fc2_start = new_subcluster_offset
                    new_fc2_end_existing = new_fc2_start + min(num_subclusters_old, num_subclusters_new)
    
                    if num_subclusters_old > 0 and num_subclusters_new > 0:
                        new_class_fc2_weight[new_fc2_start:new_fc2_end_existing, new_fc1_start:new_fc1_end] = \
                            old_class_fc2_weight[old_fc2_start:old_fc2_end, old_fc1_start:old_fc1_end]
                        new_class_fc2_bias[new_fc2_start:new_fc2_end_existing] = old_class_fc2_bias[old_fc2_start:old_fc2_end]
    
                    # Initialize additional subclusters if needed
                    num_subclusters_to_initialize = num_subclusters_new - min(num_subclusters_old, num_subclusters_new)
                    if num_subclusters_to_initialize > 0:
                        new_fc2_end = new_fc2_start + num_subclusters_new
                        new_class_fc2_weight[new_fc2_end_existing:new_fc2_end, new_fc1_start:new_fc1_end] = \
                            torch.randn(num_subclusters_to_initialize, self.hidden_dim, device=self.device)
                        new_class_fc2_bias[new_fc2_end_existing:new_fc2_end] = torch.zeros(num_subclusters_to_initialize, device=self.device)
                    else:
                        new_fc2_end = new_fc2_end_existing
    
                    print(f"DEBUG: Copied weights and biases for merged cluster {highest_ll_idx} to new cluster {new_cluster_idx}.")
    
                elif init_new_weights == "random":
                    # Initialize weights and biases randomly
                    new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = torch.randn(self.hidden_dim, self.codes_dim, device=self.device)
                    new_class_fc1_bias[new_fc1_start:new_fc1_end] = torch.zeros(self.hidden_dim, device=self.device)
    
                    # Initialize class_fc2 weights and biases
                    new_fc2_start = new_subcluster_offset
                    new_fc2_end = new_fc2_start + num_subclusters_new
                    new_class_fc2_weight[new_fc2_start:new_fc2_end, new_fc1_start:new_fc1_end] = torch.randn(num_subclusters_new, self.hidden_dim, device=self.device)
                    new_class_fc2_bias[new_fc2_start:new_fc2_end] = torch.zeros(num_subclusters_new, device=self.device)
    
                    print(f"DEBUG: Initialized weights and biases randomly for merged cluster {highest_ll_idx} to new cluster {new_cluster_idx}.")
    
                else:
                    print(f"ERROR: Unknown init_new_weights option '{init_new_weights}'.")
                    return  # Early exit due to error
    
                # Update mappings and offsets
                cluster_idx_mapping[highest_ll_idx] = new_cluster_idx
                print(f"DEBUG: Mapped merged cluster {highest_ll_idx} to new cluster {new_cluster_idx}.")
                new_cluster_idx += 1
                new_subcluster_offset += num_subclusters_new
    
            # Assign the new weights and biases
            print("DEBUG: Assigning new weights and biases to class_fc1 and class_fc2.")
            self.class_fc1.weight.data.copy_(new_class_fc1_weight)
            self.class_fc1.bias.data.copy_(new_class_fc1_bias)
            self.class_fc2.weight.data.copy_(new_class_fc2_weight)
            self.class_fc2.bias.data.copy_(new_class_fc2_bias)
    
            # Update gradient mask if necessary
            # self._update_gradient_mask()
            print("DEBUG: Updated gradient mask.")
    
        print("DEBUG: Finished updating subclustering network after merge")
    
    
    def _update_gradient_mask(self):
        print("DEBUG: Updating gradient mask for class_fc2")
        # Create a gradient mask for detaching different subclustering nets
        gradient_mask_fc2 = torch.zeros(self.hidden_dim * self.K, self.total_subclusters, device=self.device)
        print(f"DEBUG: Initialized new gradient_mask_fc2 with shape {gradient_mask_fc2.shape}")
    
        subcluster_offset = 0
        for k in range(self.K):
            num_subclusters = self.subclusters_per_cluster[k]
            start_idx = self.hidden_dim * k
            end_idx = start_idx + self.hidden_dim
            gradient_mask_fc2[start_idx:end_idx, subcluster_offset: subcluster_offset + num_subclusters] = 1
            print(f"DEBUG: Setting gradient mask for cluster {k} with {num_subclusters} subclusters (indices {subcluster_offset} to {subcluster_offset + num_subclusters -1})")
            subcluster_offset += num_subclusters
    
        print(f"DEBUG: Final gradient_mask_fc2 =\n{gradient_mask_fc2}")
        # No need to multiply weights here; the mask is applied in the gradient hook
    
        # Re-register the gradient hook with updated mask
        def grad_hook(grad):
            # Apply the mask to the gradients
            modified_grad = grad * gradient_mask_fc2.T
            return modified_grad
    
        # Remove previous hooks to avoid multiple hooks
        if self.class_fc2.weight._backward_hooks:
            self.class_fc2.weight._backward_hooks.clear()
    
        self.class_fc2.weight.register_hook(grad_hook)

    def _update_gradient_mask_v1(self):
        print("DEBUG: Updating gradient mask for class_fc2")
        # Create a gradient mask for detaching different subclustering nets
        gradient_mask_fc2 = torch.zeros(self.hidden_dim * self.K, self.total_subclusters, device=self.device)
        print(f"DEBUG: Initialized new gradient_mask_fc2 with shape {gradient_mask_fc2.shape}")
        #print("Updated gradient mask:")
        #print(gradient_mask_fc2)

        subcluster_offset = 0
        for k in range(self.K):
            num_subclusters = self.subclusters_per_cluster[k]
            start_idx = self.hidden_dim * k
            end_idx = start_idx + self.hidden_dim
            gradient_mask_fc2[start_idx:end_idx, subcluster_offset: subcluster_offset + num_subclusters] = 1
            print(f"DEBUG: Setting gradient mask for cluster {k} with {num_subclusters} subclusters")
            subcluster_offset += num_subclusters

        print(f"DEBUG: Final gradient_mask_fc2 =\n{gradient_mask_fc2}")
        self.class_fc2.weight.data *= gradient_mask_fc2.T
        print(f"DEBUG: After applying new gradient mask, class_fc2.weight.data =\n{self.class_fc2.weight.data}")

        # Re-register the gradient hook with updated mask
        def grad_hook(grad):
            #print("DEBUG: Gradient hook triggered for class_fc2.weight after mask update")
            #print(f"DEBUG: Original grad shape = {grad.shape}")
            modified_grad = grad.mul_(gradient_mask_fc2.T)
            #print(f"DEBUG: Modified grad shape = {modified_grad.shape}")
            return modified_grad

        # Remove previous hooks to avoid multiple hooks
        if self.class_fc2.weight._backward_hooks is not None:
           self.class_fc2.weight._backward_hooks.clear()
        
        self.class_fc2.weight.register_hook(grad_hook)
        #print("DEBUG: Re-registered gradient hook for class_fc2.weight")

    def update_K_split(
    self,
    split_decisions,
    init_new_weights="same",
    n_sub_list_new=None,
    accepted_subclusters=None  # NEW: to handle partial acceptance
):
        """
        Updates the subclustering network after clusters have been split.
        Now supports partial acceptance of subclusters AND variable
        numbers of sub-subclusters.
    
        Parameters
        ----------
        split_decisions : torch.BoolTensor or list[bool]
            Which main clusters are being split in the old model.
        init_new_weights : str
            How to initialize weights for newly created clusters ("same", "random", etc.).
        n_sub_list_new : list[int]
            Updated number of sub-subclusters for each newly formed cluster (the final dimension
            of the subclustering net). Each accepted subcluster i becomes a new cluster, which
            can have an arbitrary number of sub-subclusters as specified in n_sub_list_new.
        accepted_subclusters : list[list[int]] or None
            For each old main cluster k, a list of old subcluster indices that were accepted.
            Each accepted subcluster forms a NEW cluster in the subclustering net.
        """
        if n_sub_list_new is None:
            raise ValueError("n_sub_list_new must be provided to update the subclustering network.")
    
        # Keep a record of how many subclusters each old cluster had
        subclusters_per_cluster_old = self.subclusters_per_cluster.copy()
        K_old = len(subclusters_per_cluster_old)
    
        # The new subclustering net dimension is based on n_sub_list_new
        #  => number of final 'clusters' in subclustering net
        self.K = len(n_sub_list_new)  
        self.subclusters_per_cluster = n_sub_list_new  # each new cluster's sub-subcluster count
        self.total_subclusters = sum(n_sub_list_new)
    
        # Convert split_decisions to a tensor if needed
        if not isinstance(split_decisions, torch.Tensor):
            split_decisions_tensor = torch.tensor(split_decisions, dtype=torch.bool, device=self.device)
        else:
            split_decisions_tensor = split_decisions.to(self.device)
    
        with torch.no_grad():
            # 1) Save the old weights/biases
            old_class_fc1_weight = self.class_fc1.weight.data.clone()
            old_class_fc1_bias   = self.class_fc1.bias.data.clone()
            old_class_fc2_weight = self.class_fc2.weight.data.clone()
            old_class_fc2_bias   = self.class_fc2.bias.data.clone()
    
            # 2) Rebuild self.class_fc1 and self.class_fc2 with new shapes
            #    (hidden_dim*K) for the 1st layer, and total_subclusters for the 2nd layer.
            self.class_fc1 = nn.Linear(self.codes_dim, self.hidden_dim * self.K).to(self.device)
            self.class_fc2 = nn.Linear(self.hidden_dim * self.K, self.total_subclusters).to(self.device)
    
            # Allocate space for new parameter data
            new_class_fc1_weight = torch.zeros_like(self.class_fc1.weight.data)
            new_class_fc1_bias   = torch.zeros_like(self.class_fc1.bias.data)
            new_class_fc2_weight = torch.zeros_like(self.class_fc2.weight.data)
            new_class_fc2_bias   = torch.zeros_like(self.class_fc2.bias.data)
    
            # Offset trackers for reading old weights
            old_cluster_idx       = 0
            old_subcluster_offset = 0
    
            # Offset trackers for placing new weights
            new_cluster_idx       = 0
            new_subcluster_offset = 0
    
            # ------------------------------------------------------------------
            # A) First, copy over clusters that are NOT being split
            # ------------------------------------------------------------------
            for k in range(K_old):
                if not split_decisions_tensor[k]:
                    # The old cluster k remains intact with the same number of subclusters
                    num_sub_old = subclusters_per_cluster_old[k]
                    num_sub_new = num_sub_old  # unchanged if not split
    
                    # Indices for old fc1
                    old_fc1_start = old_cluster_idx * self.hidden_dim
                    old_fc1_end   = old_fc1_start + self.hidden_dim
    
                    # Indices for new fc1
                    new_fc1_start = new_cluster_idx * self.hidden_dim
                    new_fc1_end   = new_fc1_start + self.hidden_dim
    
                    # Copy old fc1 weights/bias
                    new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = \
                        old_class_fc1_weight[old_fc1_start:old_fc1_end, :]
                    new_class_fc1_bias[new_fc1_start:new_fc1_end] = \
                        old_class_fc1_bias[old_fc1_start:old_fc1_end]
    
                    # Indices for old fc2
                    old_fc2_start = old_subcluster_offset
                    old_fc2_end   = old_fc2_start + num_sub_old
    
                    # Indices for new fc2
                    new_fc2_start = new_subcluster_offset
                    new_fc2_end   = new_fc2_start + num_sub_new
    
                    # Copy old fc2 weights/bias
                    new_class_fc2_weight[new_fc2_start:new_fc2_end, new_fc1_start:new_fc1_end] = \
                        old_class_fc2_weight[old_fc2_start:old_fc2_end, old_fc1_start:old_fc1_end]
                    new_class_fc2_bias[new_fc2_start:new_fc2_end] = \
                        old_class_fc2_bias[old_fc2_start:old_fc2_end]
    
                    # Update offsets
                    old_cluster_idx       += 1
                    old_subcluster_offset += num_sub_old
                    new_cluster_idx       += 1
                    new_subcluster_offset += num_sub_new
    
            # ------------------------------------------------------------------
            # B) Process clusters that ARE being split:
            #    Each accepted subcluster => a brand NEW cluster in subclustering net
            # ------------------------------------------------------------------
            for k in range(K_old):
                if split_decisions_tensor[k]:
                    num_sub_old = subclusters_per_cluster_old[k]
    
                    # Indices for old fc1
                    old_fc1_start = old_cluster_idx * self.hidden_dim
                    old_fc1_end   = old_fc1_start + self.hidden_dim
    
                    # If partial acceptance is in use, only loop over accepted_subclusters[k]
                    if accepted_subclusters is not None:
                        sub_idx_list = accepted_subclusters[k]
                    else:
                        # If not provided, fallback to all old subclusters
                        sub_idx_list = range(num_sub_old)
    
                    # For each accepted subcluster i from old cluster k
                    for i in sub_idx_list:
                        # This subcluster i becomes *one* new cluster in the subclustering net
                        # => That new cluster can have "n_sub_list_new[new_cluster_idx]" sub-subclusters
                        num_sub_new = self.subclusters_per_cluster[new_cluster_idx]
    
                        # Indices for new fc1
                        new_fc1_start = new_cluster_idx * self.hidden_dim
                        new_fc1_end   = new_fc1_start + self.hidden_dim
    
                        # Initialize fc1 for the new cluster
                        if init_new_weights == "same":
                            # Copy the old cluster's fc1 block
                            new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = \
                                old_class_fc1_weight[old_fc1_start:old_fc1_end, :]
                            new_class_fc1_bias[new_fc1_start:new_fc1_end] = \
                                old_class_fc1_bias[old_fc1_start:old_fc1_end]
                        elif init_new_weights == "random":
                            # Random initialization
                            new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = \
                                torch.randn(self.hidden_dim, self.codes_dim, device=self.device)
                            new_class_fc1_bias[new_fc1_start:new_fc1_end] = \
                                torch.zeros(self.hidden_dim, device=self.device)
    
                        # Indices for old fc2
                        old_fc2_row_for_sub_i = old_subcluster_offset + i
                        # We'll replicate that single row 'i' for however many sub-subclusters are in the new cluster
    
                        # Indices for new fc2
                        new_fc2_start = new_subcluster_offset
                        new_fc2_end   = new_fc2_start + num_sub_new
    
                        if init_new_weights == "same":
                            # Copy the single row for subcluster i and replicate it for 'num_sub_new' sub-subclusters
                            row_i = old_class_fc2_weight[old_fc2_row_for_sub_i, old_fc1_start:old_fc1_end]
                            new_class_fc2_weight[new_fc2_start:new_fc2_end, new_fc1_start:new_fc1_end] = \
                                row_i.unsqueeze(0).repeat(num_sub_new, 1)
    
                            # The bias for subcluster i is repeated among all new sub-subclusters
                            bias_i = old_class_fc2_bias[old_fc2_row_for_sub_i]
                            new_class_fc2_bias[new_fc2_start:new_fc2_end] = bias_i
    
                        elif init_new_weights == "random":
                            # Random initialization for the new sub-subclusters
                            new_class_fc2_weight[new_fc2_start:new_fc2_end, new_fc1_start:new_fc1_end] = \
                                torch.randn(num_sub_new, self.hidden_dim, device=self.device)
                            new_class_fc2_bias[new_fc2_start:new_fc2_end] = \
                                torch.zeros(num_sub_new, device=self.device)
    
                        # Advance the new cluster offset
                        new_cluster_idx       += 1
                        new_subcluster_offset += num_sub_new
    
                    # Once done with cluster k, move the old offsets to the next old cluster
                    old_cluster_idx       += 1
                    old_subcluster_offset += num_sub_old
    
            # 3) Assign the new parameter data to self.class_fc1/fc2
            self.class_fc1.weight.data.copy_(new_class_fc1_weight)
            self.class_fc1.bias.data.copy_(new_class_fc1_bias)
            self.class_fc2.weight.data.copy_(new_class_fc2_weight)
            self.class_fc2.bias.data.copy_(new_class_fc2_bias)

    def update_K_split_N_subcluster(self, split_decisions, init_new_weights="same", n_sub_list_new=None):
        # Ensure n_sub_list_new is provided
        if n_sub_list_new is None:
            raise ValueError("n_sub_list_new must be provided to update the subclustering network.")
        
        # Save the old subclusters per cluster before updating
        subclusters_per_cluster_old = self.subclusters_per_cluster.copy()
        K_old = len(subclusters_per_cluster_old)
        
        # Update K and total_subclusters
        self.K = len(n_sub_list_new)
        self.subclusters_per_cluster = n_sub_list_new
        self.total_subclusters = sum(self.subclusters_per_cluster)
        
        with torch.no_grad():
            # Save old weights and biases
            old_class_fc1_weight = self.class_fc1.weight.data.clone()
            old_class_fc1_bias = self.class_fc1.bias.data.clone()
            old_class_fc2_weight = self.class_fc2.weight.data.clone()
            old_class_fc2_bias = self.class_fc2.bias.data.clone()
            
            # Update class_fc1 and class_fc2 with new dimensions
            self.class_fc1 = nn.Linear(self.codes_dim, self.hidden_dim * self.K).to(self.device)
            self.class_fc2 = nn.Linear(self.hidden_dim * self.K, self.total_subclusters).to(self.device)
            
            # Create new weight and bias tensors
            new_class_fc1_weight = torch.zeros_like(self.class_fc1.weight.data)
            new_class_fc1_bias = torch.zeros_like(self.class_fc1.bias.data)
            new_class_fc2_weight = torch.zeros_like(self.class_fc2.weight.data)
            new_class_fc2_bias = torch.zeros_like(self.class_fc2.bias.data)
            
            # Prepare split decisions tensor
            split_decisions_tensor = split_decisions if isinstance(split_decisions, torch.Tensor) else torch.tensor(split_decisions)
            
            # Offset trackers
            old_cluster_idx = 0
            new_cluster_idx = 0
            old_subcluster_offset = 0
            new_subcluster_offset = 0
            
            # First, process clusters not being split
            for k in range(K_old):
                if not split_decisions_tensor[k]:
                    num_subclusters_old = subclusters_per_cluster_old[k]
                    num_subclusters_new = num_subclusters_old  # Since they're not being split
                    
                    # Indices for class_fc1
                    old_fc1_start = old_cluster_idx * self.hidden_dim
                    old_fc1_end = old_fc1_start + self.hidden_dim
                    new_fc1_start = new_cluster_idx * self.hidden_dim
                    new_fc1_end = new_fc1_start + self.hidden_dim
                    
                    # Copy weights and biases for class_fc1
                    new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = \
                        old_class_fc1_weight[old_fc1_start:old_fc1_end, :]
                    new_class_fc1_bias[new_fc1_start:new_fc1_end] = \
                        old_class_fc1_bias[old_fc1_start:old_fc1_end]
                    
                    # Indices for class_fc2
                    old_fc2_start = old_subcluster_offset
                    old_fc2_end = old_fc2_start + num_subclusters_old
                    new_fc2_start = new_subcluster_offset
                    new_fc2_end = new_fc2_start + num_subclusters_new
                    
                    # Copy weights and biases for class_fc2
                    new_class_fc2_weight[new_fc2_start:new_fc2_end, new_fc1_start:new_fc1_end] = \
                        old_class_fc2_weight[old_fc2_start:old_fc2_end, old_fc1_start:old_fc1_end]
                    new_class_fc2_bias[new_fc2_start:new_fc2_end] = \
                        old_class_fc2_bias[old_fc2_start:old_fc2_end]
                    
                    # Update offsets
                    old_cluster_idx += 1
                    new_cluster_idx += 1
                    old_subcluster_offset += num_subclusters_old
                    new_subcluster_offset += num_subclusters_new
            
            # Now, process clusters being split and add their new clusters sequentially
            for k in range(K_old):
                if split_decisions_tensor[k]:
                    num_subclusters_old = subclusters_per_cluster_old[k]
                    
                    # Indices for old class_fc1
                    old_fc1_start = old_cluster_idx * self.hidden_dim
                    old_fc1_end = old_fc1_start + self.hidden_dim
                    
                    # For clusters being split, we process each subcluster
                    for i in range(num_subclusters_old):
                        # Each subcluster becomes a new cluster
                        num_subclusters_new = self.subclusters_per_cluster[new_cluster_idx]
                        
                        # Indices for new class_fc1
                        new_fc1_start = new_cluster_idx * self.hidden_dim
                        new_fc1_end = new_fc1_start + self.hidden_dim
                        
                        # Initialize weights and biases for class_fc1
                        if init_new_weights == "same":
                            # Copy the old weights and biases
                            new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = \
                                old_class_fc1_weight[old_fc1_start:old_fc1_end, :]
                            new_class_fc1_bias[new_fc1_start:new_fc1_end] = \
                                old_class_fc1_bias[old_fc1_start:old_fc1_end]
                        elif init_new_weights == "random":
                            # Initialize randomly
                            new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = \
                                torch.randn(self.hidden_dim, self.codes_dim, device=self.device)
                            new_class_fc1_bias[new_fc1_start:new_fc1_end] = \
                                torch.zeros(self.hidden_dim, device=self.device)
                        
                        # Initialize weights and biases for class_fc2
                        new_fc2_start = new_subcluster_offset
                        new_fc2_end = new_fc2_start + num_subclusters_new
                        
                        if init_new_weights == "same":
                            # Initialize by copying or interpolating old weights
                            # For simplicity, let's initialize randomly here
                            new_class_fc2_weight[new_fc2_start:new_fc2_end, new_fc1_start:new_fc1_end] = \
                                old_class_fc2_weight[old_fc2_start + i, old_fc1_start:old_fc1_end].unsqueeze(0).repeat(num_subclusters_new, 1)
                            new_class_fc2_bias[new_fc2_start:new_fc2_end] = \
                                old_class_fc2_bias[old_fc2_start + i]
                        elif init_new_weights == "random":
                            # Initialize randomly
                            new_class_fc2_weight[new_fc2_start:new_fc2_end, new_fc1_start:new_fc1_end] = \
                                torch.randn(num_subclusters_new, self.hidden_dim, device=self.device)
                            new_class_fc2_bias[new_fc2_start:new_fc2_end] = \
                                torch.zeros(num_subclusters_new, device=self.device)
                        
                        # Update offsets
                        new_cluster_idx += 1
                        new_subcluster_offset += num_subclusters_new
                    
                    # Update old offsets after processing the split cluster
                    old_cluster_idx += 1
                    old_subcluster_offset += num_subclusters_old
            
            # Assign the new weights and biases
            self.class_fc1.weight.data.copy_(new_class_fc1_weight)
            self.class_fc1.bias.data.copy_(new_class_fc1_bias)
            self.class_fc2.weight.data.copy_(new_class_fc2_weight)
            self.class_fc2.bias.data.copy_(new_class_fc2_bias)



    def _initalize_weights_merge(self, weights_list, highest_ll_loc, init_new_weight="highest_ll", num=2):
        if init_new_weight == "highest_ll":
            # keep the weights of the more likely cluster
            return weights_list[highest_ll_loc]
        elif init_new_weight == "random_choice":
            return weights_list[torch.round(torch.rand(1)).int().item()]
        elif init_new_weight == "random":
            return torch.FloatTensor(weights_list[0].shape[0], weights_list[0].shape[1]).uniform_(-1., 1).to(device=self.device)
        else:
            raise NotImplementedError
    

    def _initalize_bias_merge(self, bias_list, highest_ll, init_new_weight="highest_ll", num=2):
        if init_new_weight == "highest_ll":
            # keep the weights of the more likely cluster
            return bias_list[highest_ll]
        elif init_new_weight == "random":
            return bias_list[torch.round(torch.rand(1)).int().item()]
        else:
            raise NotImplementedError
    def log_subcluster_labels(self):
        print("Subcluster Labels:")
        for cluster_idx, sub_labels in self.subcluster_labels.items():
            cluster_label = self.cluster_labels[cluster_idx]
            print(f"Cluster '{cluster_label}' (MLP index {cluster_idx}):")
            for sub_idx, sub_label in sub_labels.items():
                print(f"  Subcluster index {sub_idx}: Label '{sub_label}'")



class Conv_Classifier(nn.Module):
    def __init__(self, hparams):
        super(Conv_Classifier, self).__init__()
        self.hparams = hparams

        raise NotImplementedError("Need to implement split merge operations!")

        # classifier
        self.class_conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.class_conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.class_conv2_drop = nn.Dropout2d()
        self.class_fc1 = nn.Linear(320, 50)
        self.class_fc2 = nn.Linear(50, hparams.init_k)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.class_conv1(x), 2))
        x = F.relu(F.max_pool2d(self.class_conv2_drop(self.class_conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.class_fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.class_fc2(x)
        return F.softmax(x, dim=1)
