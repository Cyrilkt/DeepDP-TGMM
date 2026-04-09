#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        if weights_fc1 is not None:
            self.class_fc1.weight.data = weights_fc1
        if weights_fc2 is not None:
            self.class_fc2.weight.data = weights_fc2
        if bias_fc1 is not None:
            self.class_fc1.bias.data = bias_fc1
        if bias_fc2 is not None:
            self.class_fc2.bias.data = bias_fc2

        self.softmax_norm = hparams.softmax_norm

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
    

    def update_K_split(self, split_decisions, init_new_weights="same", subclusters_nets=None, n_sub_list=None):
        """
        Generalized update_K_split function to handle variable number of subclusters per cluster.
    
        Parameters:
        - split_decisions: List of booleans indicating whether to split each cluster.
        - init_new_weights: Method for initializing new weights ("same", "random").
        - subclusters_nets: Subclustering network (if any).
        - n_sub_list: List containing the number of subclusters per cluster.
        """
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
    
            weights_not_split = class_fc2.weight.data[not_split_indices, :]
            weights_split = class_fc2.weight.data[split_indices, :]
    
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


    

    def update_K_merge(self, merge_decisions, pairs_to_merge, highest_ll, init_new_weights="same"):
        """ Update the clustering net after a merge decision was made

        Args:
            merge_decisions (torch.tensor): a list of K booleans indicating whether to a cluster should be merged or not
            pairs_to_merge ([type]): a list of lists, which list contains the indices of two clusters to merge
            init_new_weights (str, optional): How to initialize the weights of the new weights of the merged cluster. Defaults to "same".
                "same" uses the weights of the cluster with the highest loglikelihood, "random" uses random weights.
            highest_ll ([type]): a list of the indices of the clusters with the highest log likelihood for each pair.

        Description:
            We will delete the weights of the two merged clusters, and append (to the end) the weights of the newly merged clusters
        """

        self.k -= len(highest_ll)

        with torch.no_grad():
            class_fc2 = nn.Linear(self.last_dim, self.k)

            # Adjust weights
            weights_not_merged = self.class_fc2.weight.data[torch.logical_not(merge_decisions), :]
            weights_merged = self.class_fc2.weight.data[merge_decisions, :]
            new_weights = self._initalize_weights_merge(
                weights_merged, merge_decisions, highest_ll, init_new_weight=init_new_weights
            )

            class_fc2.weight.data = torch.cat(
                [weights_not_merged, new_weights]
            )

            # Adjust bias
            bias_not_merged = self.class_fc2.bias.data[torch.logical_not(merge_decisions)]
            bias_merged = self.class_fc2.bias.data[merge_decisions]
            new_bias = self._initalize_bias_merge(bias_merged, merge_decisions, highest_ll, init_new_weight=init_new_weights)
            class_fc2.bias.data = torch.cat([bias_not_merged, new_bias])

            self.class_fc2 = class_fc2
    
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
    def __init__(self, hparams, codes_dim=320, k=None):
        super(MLP_Classifier, self).__init__()
        if k is None:
            self.K = hparams.init_k
        else:
            self.K = k

        self.codes_dim = codes_dim
        self.hparams = hparams
        self.hidden_dim = 50
        self.softmax_norm = self.hparams.subcluster_softmax_norm

        # the subclustering net will be a stacked version of the clustering net
        self.class_fc1 = nn.Linear(self.codes_dim * self.K, self.hidden_dim * self.K)
        self.class_fc2 = nn.Linear(self.hidden_dim * self.K, 2 * self.K)

        gradient_mask_fc1 = torch.ones(self.codes_dim * self.K, self.hidden_dim * self.K)
        gradient_mask_fc2 = torch.ones(self.hidden_dim * self.K, 2 * self.K)
        # detach different subclustering nets - zeroing out the weights connecting between different subnets
        # and also zero their gradient
        for k in range(self.K):
            # row are the output neurons and columns are of the input ones
            # before
            self.class_fc1.weight.data[self.hidden_dim * k: self.hidden_dim * (k + 1), :self.codes_dim * k] = 0
            gradient_mask_fc1[self.hidden_dim * k: self.hidden_dim * (k + 1), :self.codes_dim * k] = 0
            self.class_fc2.weight.data[2 * k: 2 * (k + 1), :self.hidden_dim * k] = 0
            gradient_mask_fc2[2 * k: 2 * (k + 1), :self.hidden_dim * k] = 0
            # after
            self.class_fc1.weight.data[self.hidden_dim * k: self.hidden_dim * (k + 1), :self.codes_dim * (k + 1)] = 0
            gradient_mask_fc1[self.hidden_dim * k: self.hidden_dim * (k + 1), :self.codes_dim * (k + 1)] = 0
            self.class_fc2.weight.data[2 * k: 2 * (k + 1), :self.hidden_dim * (k + 1)] = 0
            gradient_mask_fc2[2 * k: 2 * (k + 1), :self.hidden_dim * (k + 1)] = 0

        self.class_fc1.weight.register_hook(lambda grad: grad.mul_(gradient_mask_fc1))
        self.class_fc2.weight.register_hook(lambda grad: grad.mul_(gradient_mask_fc2))
        # weights are zero and their grad will always be 0 so won't change

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

        # Total number of subclusters is the sum of the list
        self.total_subclusters = sum(subclusters_per_cluster)

        self.codes_dim = codes_dim
        self.hparams = hparams
        self.hidden_dim = 50
        self.softmax_norm = self.hparams.softmax_norm
        self.device = "cuda" if torch.cuda.is_available() and hparams.gpus is not None else "cpu"

        # the subclustering net will be a stacked version of the clustering net
        self.class_fc1 = nn.Linear(self.codes_dim, self.hidden_dim * self.K)
        self.class_fc2 = nn.Linear(self.hidden_dim * self.K, self.total_subclusters)

        # Create a gradient mask for detaching different subclustering nets
        gradient_mask_fc2 = torch.zeros(self.hidden_dim * self.K, self.total_subclusters)

        subcluster_offset = 0  # To handle different numbers of subclusters per cluster
        for k in range(self.K):
            num_subclusters = subclusters_per_cluster[k]
            gradient_mask_fc2[self.hidden_dim * k:self.hidden_dim * (k + 1), subcluster_offset: subcluster_offset + num_subclusters] = 1
            subcluster_offset += num_subclusters

        self.class_fc2.weight.data *= gradient_mask_fc2.T
        self.class_fc2.weight.register_hook(lambda grad: grad.mul_(gradient_mask_fc2.T.to(device=self.device)))

        # weights are zero and their grad will always be 0 so won't change

    def forward(self, X):
        # Note that there is no softmax here
        X = F.relu(self.class_fc1(X))
        X = self.class_fc2(X)
        return X
    
    def _initalize_weights_split(self, weights_old, init_new_weights, num_new_subclusters):
        """
        Initialize new weights for subclusters after splitting.
    
        Parameters:
        - weights_old: Tensor of old weights (num_old_subclusters x hidden_dim)
        - init_new_weights: Method for initializing new weights ("same", "random").
        - num_new_subclusters: Number of new subclusters to initialize.
    
        Returns:
        - new_weights: Tensor of new weights (num_new_subclusters x hidden_dim).
        """
        if init_new_weights == "same":
            # Duplicate the old weights to match the number of new subclusters
            old_num_subclusters = weights_old.shape[0]
            if old_num_subclusters == num_new_subclusters:
                # Same number of subclusters, just copy
                new_weights = weights_old.clone()
            else:
                repeats = num_new_subclusters // old_num_subclusters
                remainder = num_new_subclusters % old_num_subclusters
                new_weights = weights_old.repeat(repeats, 1)
                if remainder > 0:
                    new_weights = torch.cat([new_weights, weights_old[:remainder]], dim=0)
        elif init_new_weights == "random":
            # Initialize randomly
            new_weights = torch.randn(num_new_subclusters, weights_old.shape[1], device=self.device)
        else:
            raise NotImplementedError("Unknown init_new_weights method: {}".format(init_new_weights))
        return new_weights
    
    def _initalize_bias_split(self, biases_old, init_new_weights, num_new_subclusters):
        """
        Initialize new biases for subclusters after splitting.
    
        Parameters:
        - biases_old: Tensor of old biases (num_old_subclusters).
        - init_new_weights: Method for initializing new biases ("same", "random").
        - num_new_subclusters: Number of new subclusters to initialize.
    
        Returns:
        - new_biases: Tensor of new biases (num_new_subclusters).
        """
        if init_new_weights == "same":
            # Duplicate the old biases to match the number of new subclusters
            old_num_subclusters = biases_old.shape[0]
            if old_num_subclusters == num_new_subclusters:
                # Same number of subclusters, just copy
                new_biases = biases_old.clone()
            else:
                repeats = num_new_subclusters // old_num_subclusters
                remainder = num_new_subclusters % old_num_subclusters
                new_biases = biases_old.repeat(repeats)
                if remainder > 0:
                    new_biases = torch.cat([new_biases, biases_old[:remainder]], dim=0)
        elif init_new_weights == "random":
            # Initialize biases to zero
            new_biases = torch.zeros(num_new_subclusters, device=self.device)
        else:
            raise NotImplementedError("Unknown init_new_weights method: {}".format(init_new_weights))
        return new_biases


    

    def update_K_merge(self, merge_decisions, pairs_to_merge, highest_ll, init_new_weights="highest_ll"):
        """ Update the clustering net after a merge decision was made

        Args:
            merge_decisions (torch.tensor): a list of K booleans indicating whether to a cluster should be merged or not
            pairs_to_merge ([type]): a list of lists, which list contains the indices of two clusters to merge
            init_new_weights (str, optional): How to initialize the weights of the new weights of the merged cluster. Defaults to "same".
                "same" uses the weights of the cluster with the highest loglikelihood, "random" uses random weights.
            highest_ll ([type]): a list of the indices of the clusters with the highest log likelihood for each pair.

        Description:
            We will delete the weights of the two merged clusters, and append (to the end) the weights of the newly merged clusters
        """

        class_fc1 = self.class_fc1
        class_fc2 = self.class_fc2
        mus_ind_not_merged = torch.nonzero(torch.logical_not(torch.tensor(merge_decisions)), as_tuple=False)
        self.K -= len(highest_ll)

        with torch.no_grad():
            self.class_fc1 = nn.Linear(self.codes_dim, self.hidden_dim * self.K)
            self.class_fc2 = nn.Linear(self.hidden_dim * self.K, 2 * self.K)

            # Adjust weights
            fc1_weights_not_merged = class_fc1.weight.data[torch.logical_not(torch.tensor(merge_decisions)).repeat_interleave(self.hidden_dim), :]
            fc1_new_weights = []
            fc1_new_bias = []
            fc2_new_bias = []
            for merge_pair, highest_ll_k in zip(pairs_to_merge, highest_ll):
                fc1_weights_merged = [
                    class_fc1.weight.data[k * self.hidden_dim: (k + 1) * self.hidden_dim, :] for k in merge_pair]
                fc1_new_weights.append(self._initalize_weights_merge(
                    fc1_weights_merged, (torch.tensor(highest_ll_k) == merge_pair[1]).item(), init_new_weight=init_new_weights
                ))

                fc1_bias_merged = [
                    class_fc1.bias.data[k * self.hidden_dim: (k + 1) * self.hidden_dim] for k in merge_pair]
                fc1_new_bias.append(self._initalize_weights_merge(
                    fc1_bias_merged, (torch.tensor(highest_ll_k) == merge_pair[1]).item(), init_new_weight=init_new_weights
                ))
            fc1_new_weights = torch.cat(fc1_new_weights)
            fc1_new_bias = torch.cat(fc1_new_bias)

            self.class_fc1.weight.data = torch.cat(
                [fc1_weights_not_merged, fc1_new_weights]
            )

            self.class_fc2.weight.data.fill_(0)
            gradient_mask_fc2 = torch.zeros(self.hidden_dim * self.K, 2 * self.K)
            for i, k in enumerate(mus_ind_not_merged):
                # i is the new index of the cluster and k is the old one
                self.class_fc2.weight.data[2 * i: 2*(i + 1), self.hidden_dim * i: self.hidden_dim * (i+1)] =\
                    class_fc2.weight.data[2 * k: 2*(k + 1), self.hidden_dim * k: self.hidden_dim * (k+1)]
                gradient_mask_fc2[self.hidden_dim * i:self.hidden_dim * (i + 1), 2 * i: 2 * (i + 1)] = 1
            for j, (merge_pair, highest_ll_k) in enumerate(zip(pairs_to_merge, highest_ll)):
                # j + len(mus_ind_not_split) is the new index and k is the old one. We use interleave to create 2 new clusters for each split cluster
                i = j + len(mus_ind_not_merged)
                weights = [class_fc2.weight.data[2 * k: 2*(k + 1), self.hidden_dim * k: self.hidden_dim * (k+1)] for k in merge_pair]
                weights = self._initalize_weights_merge(weights, (torch.tensor(highest_ll_k) == merge_pair[1]).item(), init_new_weights)
                bias = [class_fc2.bias.data[2 * k: 2*(k + 1)] for k in merge_pair]
                bias = self._initalize_weights_merge(bias, (torch.tensor(highest_ll_k) == merge_pair[1]).item(), init_new_weights)
                fc2_new_bias.append(bias)
                self.class_fc2.weight.data[2 * i: 2*(i + 1), self.hidden_dim * i: self.hidden_dim * (i+1)] = weights
                gradient_mask_fc2[self.hidden_dim * i:self.hidden_dim * (i + 1), 2 * i: 2 * (i + 1)] = 1

            self.class_fc2.weight.register_hook(lambda grad: grad.mul_(gradient_mask_fc2.T.to(device=self.device)))
            fc2_new_bias = torch.cat(fc2_new_bias)

            # Adjust bias
            fc1_bias_not_merged = class_fc1.bias.data[torch.logical_not(merge_decisions).repeat_interleave(self.hidden_dim)]
            fc2_bias_not_merged = class_fc2.bias.data[torch.logical_not(merge_decisions).repeat_interleave(2)]

            self.class_fc1.bias.data = torch.cat([fc1_bias_not_merged, fc1_new_bias])
            self.class_fc2.bias.data = torch.cat([fc2_bias_not_merged, fc2_new_bias])
            self.class_fc1.to(device=self.device)
            self.class_fc2.to(device=self.device)

            del class_fc1, class_fc2
    def _update_gradient_mask(self):
        """
        Updates the gradient mask for the subclustering network to ensure that weights are only updated for their respective clusters.
        """
        # Create a gradient mask for detaching different subclustering nets
        gradient_mask_fc2 = torch.zeros(self.hidden_dim * self.K, self.total_subclusters, device=self.device)
    
        subcluster_offset = 0
        for k in range(self.K):
            num_subclusters = self.subclusters_per_cluster[k]
            start_idx = self.hidden_dim * k
            end_idx = start_idx + self.hidden_dim
            gradient_mask_fc2[start_idx:end_idx, subcluster_offset: subcluster_offset + num_subclusters] = 1
            subcluster_offset += num_subclusters
    
        # Apply the gradient mask to the weights
        self.class_fc2.weight.data *= gradient_mask_fc2.T
        self.class_fc2.weight.register_hook(lambda grad: grad.mul_(gradient_mask_fc2.T))

    def update_K_split(self, split_decisions, init_new_weights="same", n_sub_list_new=None):
        """
        Updates the subclustering network after clusters have been split.
    
        Parameters:
        - split_decisions: Tensor of booleans indicating which clusters to split.
        - init_new_weights: Method for initializing new weights ("same", "random").
        - n_sub_list_new: List containing the updated number of subclusters per cluster after splitting.
        """
        # Ensure n_sub_list_new is provided
        if n_sub_list_new is None:
            raise ValueError("n_sub_list_new must be provided to update the subclustering network.")
    
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
    
            # Offset trackers
            old_cluster_idx = 0
            new_cluster_idx = 0
            old_subcluster_offset = 0
            new_subcluster_offset = 0
    
            split_decisions_tensor = split_decisions if isinstance(split_decisions, torch.Tensor) else torch.tensor(split_decisions)
            split_indices = split_decisions_tensor.nonzero(as_tuple=False).squeeze().tolist()
            if isinstance(split_indices, int):
                split_indices = [split_indices]
    
            # Iterate over clusters
            for k in range(len(split_decisions_tensor)):
                num_subclusters_old = self.subclusters_per_cluster[k] if hasattr(self, 'subclusters_per_cluster') else 2  # Default to 2
                num_subclusters_new = n_sub_list_new[k]
    
                old_fc1_start = old_cluster_idx * self.hidden_dim
                old_fc1_end = old_fc1_start + self.hidden_dim
                new_fc1_start = new_cluster_idx * self.hidden_dim
                new_fc1_end = new_fc1_start + self.hidden_dim
    
                if not split_decisions_tensor[k]:
                    # Cluster not being split
                    # Copy weights and biases for class_fc1
                    new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = old_class_fc1_weight[old_fc1_start:old_fc1_end, :]
                    new_class_fc1_bias[new_fc1_start:new_fc1_end] = old_class_fc1_bias[old_fc1_start:old_fc1_end]
    
                    # Copy weights and biases for class_fc2
                    old_fc2_start = old_subcluster_offset
                    old_fc2_end = old_fc2_start + num_subclusters_old
                    new_fc2_start = new_subcluster_offset
                    new_fc2_end = new_fc2_start + num_subclusters_new
    
                    new_class_fc2_weight[new_fc2_start:new_fc2_end, new_fc1_start:new_fc1_end] = \
                        old_class_fc2_weight[old_fc2_start:old_fc2_end, old_fc1_start:old_fc1_end]
                    new_class_fc2_bias[new_fc2_start:new_fc2_end] = old_class_fc2_bias[old_fc2_start:old_fc2_end]
    
                    # Update offsets
                    old_cluster_idx += 1
                    new_cluster_idx += 1
                    old_subcluster_offset += num_subclusters_old
                    new_subcluster_offset += num_subclusters_new
                else:
                    # Cluster is being split
                    # Initialize weights and biases for class_fc1
                    if init_new_weights == "same":
                        # Copy the old weights and biases
                        new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = old_class_fc1_weight[old_fc1_start:old_fc1_end, :]
                        new_class_fc1_bias[new_fc1_start:new_fc1_end] = old_class_fc1_bias[old_fc1_start:old_fc1_end]
                    elif init_new_weights == "random":
                        # Initialize randomly
                        new_class_fc1_weight[new_fc1_start:new_fc1_end, :] = torch.randn(self.hidden_dim, self.codes_dim, device=self.device)
                        new_class_fc1_bias[new_fc1_start:new_fc1_end] = torch.zeros(self.hidden_dim, device=self.device)
    
                    # Initialize weights and biases for class_fc2
                    old_fc2_start = old_subcluster_offset
                    old_fc2_end = old_fc2_start + num_subclusters_old
                    new_fc2_start = new_subcluster_offset
                    new_fc2_end = new_fc2_start + num_subclusters_new
    
                    weights_old = old_class_fc2_weight[old_fc2_start:old_fc2_end, old_fc1_start:old_fc1_end]
                    biases_old = old_class_fc2_bias[old_fc2_start:old_fc2_end]
    
                    new_weights = self._initalize_weights_split(weights_old, init_new_weights, num_subclusters_new)
                    new_biases = self._initalize_bias_split(biases_old, init_new_weights, num_subclusters_new)
    
                    new_class_fc2_weight[new_fc2_start:new_fc2_end, new_fc1_start:new_fc1_end] = new_weights
                    new_class_fc2_bias[new_fc2_start:new_fc2_end] = new_biases
    
                    # Update offsets
                    old_cluster_idx += 1
                    new_cluster_idx += 1
                    old_subcluster_offset += num_subclusters_old
                    new_subcluster_offset += num_subclusters_new
    
            # Assign the new weights and biases
            self.class_fc1.weight.data.copy_(new_class_fc1_weight)
            self.class_fc1.bias.data.copy_(new_class_fc1_bias)
            self.class_fc2.weight.data.copy_(new_class_fc2_weight)
            self.class_fc2.bias.data.copy_(new_class_fc2_bias)
    
            # Update the gradient mask for detaching different subclustering nets
            self._update_gradient_mask()


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
