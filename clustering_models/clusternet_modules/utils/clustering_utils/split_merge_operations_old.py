#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
import numpy as np
from math import lgamma
import math

from kmeans_pytorch import kmeans as GPU_KMeans
import torch.distributions as dist
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
import itertools
from src.clustering_models.clusternet_modules.utils.clustering_utils.clustering_operations import (
    _create_subclusters,
    compute_data_covs_soft_assignment,
    init_mus_and_covs_sub,
    comp_subclusters_params_min_dist,
    rotate_vector_a_to_b,
    compute_merged_mean,
    Log_mapping,
    Exp_mapping,
    KarcherMean,
    compute_merged_mean_karcher
)




def log_Hastings_ratio_split(
    alpha, N_s_list, log_ll_k_s_list, log_ll_k, split_prob,
    qbefore_split, qafter_split
):
    """
    Computes the log Hastings ratio for a split with a variable number of valid subclusters.
    Also prints important intermediate components for debugging.
    """
    N_k = sum(N_s_list)
    n_sub_k = len(N_s_list)  # Number of valid subclusters

    # Proceed if all subclusters are non-empty
    if all(N_s > 0 for N_s in N_s_list):
        # Convert scalar inputs to tensors
        alpha_tensor = torch.tensor(alpha, dtype=torch.float)
        N_s_tensors = [torch.tensor(N_s, dtype=torch.float) for N_s in N_s_list]
        # Assuming log_ll_k_s_list elements are already torch tensors
        log_ll_k_s_tensors = [x for x in log_ll_k_s_list]
        N_k_tensor = torch.tensor(N_k, dtype=torch.float)

        # Gamma function terms: compute log Gamma for each subcluster size and for the combined cluster size
        log_gamma_N_s = [torch.lgamma(x) for x in N_s_tensors]
        log_gamma_N_k = torch.lgamma(N_k_tensor)

        # Print important component values
        print("----- Debug Info: Split Score Components -----")
        print("Alpha:", alpha_tensor.item())
        print("Subcluster sizes (N_s_list):", N_s_list)
        print("Total count (N_k):", N_k)
        print("Number of subclusters (n_sub_k):", n_sub_k)
        print("Log Gamma for each subcluster (log_gamma_N_s):", [x.item() for x in log_gamma_N_s])
        print("Log Gamma for combined N_k (log_gamma_N_k):", log_gamma_N_k.item())
        print("Log likelihood for each subcluster (log_ll_k_s):", [x.item() for x in log_ll_k_s_tensors])
        # qafter_split and qbefore_split could be tensors or scalars
        if isinstance(qafter_split, torch.Tensor):
            print("qafter_split:", qafter_split.item())
        else:
            print("qafter_split:", qafter_split)
        if isinstance(qbefore_split, torch.Tensor):
            print("qbefore_split:", qbefore_split.item())
        else:
            print("qbefore_split:", qbefore_split)

        # Log joint after split:
        logJointAfter = (
            (n_sub_k - 1) * torch.log(alpha_tensor)
            + sum(log_gamma_N_s)
            + sum(log_ll_k_s_tensors)
            + qafter_split
        )
        print("Log Joint After split (logJointAfter):", logJointAfter.item())

        # Log joint before split:
        logJointBefore = (
            log_gamma_N_k
            + log_ll_k
            + qbefore_split
        )
        print("Log Joint Before split (logJointBefore):", logJointBefore.item())

        # Hastings ratio as the difference between log joint probabilities
        H = logJointAfter - logJointBefore
        print("[Hastings Ratio] H =", H.item())

        # If no user-defined split_prob, interpret it as e^H (i.e., the probability of splitting)
        if split_prob is None:
            split_prob = torch.exp(H)
            print("Calculated split probability (exp(H)):", split_prob.item())
        else:
            if isinstance(split_prob, torch.Tensor):
                print("User-provided split probability:", split_prob.item())
            else:
                print("User-provided split probability:", split_prob)
    else:
        print("[Hastings] One of the subclusters ended up empty.")
        H = torch.zeros(1)
        split_prob = torch.tensor(0.0)

    # Decide if a split should occur based on the Hastings ratio or a random draw weighed by split_prob.
    # A positive H or a random number less than split_prob will trigger the split.
    should_split = bool(
        H > 0 or (split_prob is not None and split_prob > torch.rand(1))
    )
    print("Decision: should_split =", should_split)
    return should_split

def log_Hastings_ratio_split_todecoch(
    alpha, N_s_list, log_ll_k_s_list, log_ll_k, split_prob,
    qbefore_split, qafter_split
):
    """
    Computes the log Hastings ratio for a split with a variable number of valid subclusters.
    """

    N_k = sum(N_s_list)
    n_sub_k = len(N_s_list)  # Number of valid subclusters

    # If no subcluster is empty (by construction, we've already excluded them), we proceed
    if all(N_s > 0 for N_s in N_s_list):
        alpha_tensor = torch.tensor(alpha, dtype=torch.float)
        N_s_tensors = [torch.tensor(N_s, dtype=torch.float) for N_s in N_s_list]
        log_ll_k_s_tensors = [x for x in log_ll_k_s_list]
        N_k_tensor = torch.tensor(N_k, dtype=torch.float)

        # Gamma function terms
        log_gamma_N_s = [torch.lgamma(x) for x in N_s_tensors]
        log_gamma_N_k = torch.lgamma(N_k_tensor)

        # Log joint after split
        logJointAfter = (
            (n_sub_k - 1) * torch.log(alpha_tensor)
            + sum(log_gamma_N_s)
            + sum(log_ll_k_s_tensors)
            + qafter_split
        )

        # Log joint before split
        logJointBefore = (
            log_gamma_N_k
            + log_ll_k
            + qbefore_split
        )

        # Hastings ratio
        H = logJointAfter - logJointBefore

        print(f"[Hastings] N_s_list={N_s_list}, H={H.item():.4f}")

        # If no user-defined split_prob, interpret it as e^H
        if split_prob is None:
            split_prob = torch.exp(H)

    else:
        # If any subcluster is empty, set ratio=0 => no split
        print("[Hastings] One of the subclusters ended up empty.")
        H = torch.zeros(1)
        split_prob = torch.tensor(0.0)

    should_split = bool(
        H > 0 or (split_prob is not None and split_prob > torch.rand(1))
    )
    return should_split


def log_Hastings_ratio_split_N_subcluster(
    alpha, N_s_list, log_ll_k_s_list, log_ll_k, split_prob, qbefore_split, qafter_split
):
    """
    Computes the log Hastings ratio for a split with a variable number of subclusters.

    Parameters:
    - alpha: Concentration parameter for the Dirichlet Process.
    - N_s_list: List of data point counts for each subcluster.
    - log_ll_k_s_list: List of log marginal likelihoods for each subcluster.
    - log_ll_k: Log marginal likelihood for the main cluster before split.
    - split_prob: Probability threshold for splitting (can be None).
    - qbefore_split: Log of the prior predictive for the main cluster before split.
    - qafter_split: Sum of logs of prior predictives for each subcluster after split.
    """
    # Total number of data points in cluster k
    N_k = sum(N_s_list)
    n_sub_k = len(N_s_list)  # Number of subclusters

    if all(N_s > 0 for N_s in N_s_list):
        # All subclusters have data points
        alpha_tensor = torch.tensor(alpha, dtype=torch.float)
        N_s_tensors = [torch.tensor(N_s, dtype=torch.float) for N_s in N_s_list]
        log_ll_k_s_tensors = [log_ll_k_s for log_ll_k_s in log_ll_k_s_list]
        N_k_tensor = torch.tensor(N_k, dtype=torch.float)

        # Compute log gamma functions
        log_gamma_N_s = [torch.lgamma(N_s_tensor) for N_s_tensor in N_s_tensors]
        log_gamma_N_k = torch.lgamma(N_k_tensor)

        # Compute log joint probability after split
        logJointAfter = (
            (n_sub_k - 1) * torch.log(alpha_tensor)
            + sum(log_gamma_N_s)
            + sum(log_ll_k_s_tensors)
            + qafter_split  # Sum of q(µ_s | x_s, z_s) for all subclusters
        )

        # Compute log joint probability before split
        logJointBefore = log_gamma_N_k + log_ll_k + qbefore_split

        # Compute Hastings ratio H
        H = logJointAfter - logJointBefore

        # Debug statements
        print("Entering log_Hastings_ratio_split function")
        print("Alpha:", alpha)
        print("N_s_list:", N_s_list)
        print("Log Likelihoods per Subcluster:", [ll.item() for ll in log_ll_k_s_tensors])
        print("Log Likelihood K:", log_ll_k.item())
        print("Total N_k:", N_k)
        print("LogJointAfter:", logJointAfter.item())
        print("LogJointBefore:", logJointBefore.item())
        print("H (Hastings Ratio):", H.item())

        # Determine split probability
        if split_prob is None:
            split_prob = torch.exp(H)
        print("Split Probability (adjusted):", split_prob.item())
    else:
        # If any subcluster is empty, cannot proceed with split
        print("One of the subclusters is empty")
        H = torch.zeros(1)
        split_prob = torch.tensor(0.0)

    # Decide whether to split based on Hastings ratio
    should_split = bool(
        H > 0 or (split_prob is not None and split_prob > torch.rand(1))
    )
    print("Should Split:", should_split)

    return should_split

## NOTE POUR LE PAPIER : LE HASTING RATIO CALCUL2 ICI EST EXACTEMENT LE MEME QUE CELUI PRESENT
## DANS LE PAPIER DE STRAUB (la deterministic rsplit tandis que rmerge Eq 20 est la randomize rmerge Eq 24)
def log_Hastings_ratio_split_2sub(
    alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, split_prob, qrandomproposal
):
    """Computes the log Hastings ratio for a split with detailed debug information."""
    print("Entering log_Hastings_ratio_split function")
    
    # Debug: Initial parameter values
    print("Alpha:", alpha)
    print("N_k_1:", N_k_1)
    print("N_k_2:", N_k_2)
    print("Log Likelihood K_1:", log_ll_k_1)
    print("Log Likelihood K_2:", log_ll_k_2)
    print("Log Likelihood K:", log_ll_k)
    print("Split Probability:", split_prob)
    print("QRandomProposal:", qrandomproposal)
    
    # Calculate total number of points
    N_k = N_k_1 + N_k_2
    print("N_k (Total Points):", N_k)
    
    if N_k_2 > 0 and N_k_1 > 0:
        # Debug: Check if both subclusters have points
        print("Both subclusters have points")
        
        qAfter = torch.tensor(0.0)
        
        # Convert to tensors
        alpha_tensor = torch.tensor(alpha, dtype=torch.float)
        N_k_1_tensor = torch.tensor(N_k_1, dtype=torch.float)
        N_k_2_tensor = torch.tensor(N_k_2, dtype=torch.float)
        N_k_tensor = torch.tensor(N_k, dtype=torch.float)
        
        # Calculate the log joint probability after and before
        logJointAfter = (
            torch.log(alpha_tensor) 
            + torch.lgamma(N_k_1_tensor) 
            + torch.lgamma(N_k_2_tensor) 
            + log_ll_k_1 
            + log_ll_k_2
        )
        print("LogJointAfter:", logJointAfter.item())
        
        logJointBefore = torch.lgamma(N_k_tensor) + log_ll_k
        print("LogJointBefore:", logJointBefore.item())
        #logJointBefore=torch.Tensor([1])
        
        # Hastings Ratio
        H = logJointAfter - logJointBefore + qrandomproposal - qAfter
        print("H (Hastings Ratio):", H.item())
        
        # Determine split probability
        if split_prob is None:
            split_prob = torch.exp(H)
        print("Split Probability (adjusted):", split_prob.item())
    
    else:
        # If any subcluster is empty
        print("One of the subclusters is empty")
        H = torch.zeros(1)
        split_prob = 0
        
    # Condition to perform a split
    should_split = bool(
        H > 0 or (split_prob is not None and split_prob > torch.rand(1))
    )
    print("Should Split:", should_split)
    
    # Return the decision to split or not
    return should_split

def log_Hastings_ratio_split8dpm(
    alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, split_prob,qrandomproposal
):
    """This function computes the log Hastings ratio for a split.

    Args:
        alpha ([float]): The alpha hyperparameter
        N_k_1 ([int]): Number of points assigned to the first subcluster
        N_k_2 ([int]): Number of points assigned to the second subcluster
        log_ll_k_1 ([float]): The log likelihood of the points in the first subcluster
        log_ll_k_2 ([float]): The log likelihood of the points in the second subcluster
        log_ll_k ([float]): The log likelihood of the points in the second subcluster
        split_prob ([type]): Probability to split a cluster even if the Hastings' ratio is not > 1

        Returns a boolean indicating whether to perform a split
    """
    N_k = N_k_1 + N_k_2
    if N_k_2 > 0 and N_k_1 > 0:
        # each subcluster is not empty
        H = (
            np.log(alpha) + lgamma(N_k_1) + log_ll_k_1 + lgamma(N_k_2) + log_ll_k_2
        ) - (lgamma(N_k) + log_ll_k)
        split_prob = split_prob or torch.exp(H)
    else:
        H = torch.zeros(1)
        split_prob = 0

    print(f'Alpha: {alpha}')
    print(f'N_k_1: {N_k_1}')
    print(f'N_k_2: {N_k_2}')
    print(f'Log Likelihood K_1: {log_ll_k_1}')
    print(f'Log Likelihood K_2: {log_ll_k_2}')
    print(f'Log Likelihood K: {log_ll_k}')
    
    if split_prob is not None:
        print(f'Split Probability: {split_prob.item()}')
    else:
        print('Split Probability: None')
    
    print(f'Hastings Ratio for Split: {H.item()}')
    print("bool(H > 0 or (split_prob is not None and split_prob > torch.rand(1)))",bool(H > 0 or (split_prob is not None and split_prob > torch.rand(1))))
    # if Hastings ratio > 1 (or 0 in log space) perform split, if not, toss a coin
    return bool(H > 0 or (split_prob is not None and split_prob > torch.rand(1)))





#Utilise la randomize rmerge
def log_Hastings_ratio_merge_2sub(
    alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, merge_prob, q_RandomParamProposalt_k_1, q_RandomParamProposalt_k_2, q_RandomParamProposalt_merge):
    # Use log for overflows
    print(f'Log Likelihood K_1: {log_ll_k_1}')
    print(f'Log Likelihood K_2: {log_ll_k_2}')
    print(f'Log Likelihood K: {log_ll_k}')
    print('INTO log hasting merge')
    print(f'alpha: {alpha}')
    print(f'N_k_1: {N_k_1}')
    print(f'N_k_2: {N_k_2}')
    
    if N_k_1 == 0:
        lgamma_1 = 0
    else:
        lgamma_1 = torch.lgamma(torch.tensor(N_k_1))
    print(f'lgamma_1: {lgamma_1}')

    if N_k_2 == 0:
        lgamma_2 = 0
    else:
        lgamma_2 = torch.lgamma(torch.tensor(N_k_2))
    print(f'lgamma_2: {lgamma_2}')

    # Hastings ratio in log space
    N_k = N_k_1 + N_k_2
    print(f'N_k: {N_k}')

    alpha_tensor = torch.tensor(alpha, dtype=torch.float)
    N_k_1_tensor = torch.tensor(N_k_1, dtype=torch.float)
    N_k_2_tensor = torch.tensor(N_k_2, dtype=torch.float)
    N_k_tensor = torch.tensor(N_k, dtype=torch.float)

    if N_k > 0:
        logJointAfter = (
            log_ll_k +
            torch.lgamma(alpha_tensor) +
            torch.lgamma(N_k_1_tensor + N_k_2_tensor) +
            torch.lgamma(0.5 * alpha + N_k_1_tensor) +
            torch.lgamma(0.5 * alpha + N_k_2_tensor)
        )
        
        logJointBefore = (
            log_ll_k_1 +
            log_ll_k_2 +
            torch.log(alpha_tensor) +
            2 * torch.lgamma(0.5 * alpha_tensor) +
            torch.lgamma(alpha_tensor + N_k_1_tensor + N_k_2_tensor) +
            torch.lgamma(N_k_1_tensor) +
            torch.lgamma(N_k_2_tensor)
        )
        
        print(f'logJointAfter: {logJointAfter}')
        print(f'logJointBefore: {logJointBefore}')

        qBefore = q_RandomParamProposalt_k_1 + q_RandomParamProposalt_k_2
        qAfter = q_RandomParamProposalt_merge

        print(f'qBefore: {qBefore}')
        print(f'qAfter: {qAfter}')

        H = logJointAfter - logJointBefore + qBefore - qAfter

    else:
        H = torch.ones(1)

    print(f'H: {H}')
    print(f'merge_prob: {merge_prob}')

    merge_prob = merge_prob or torch.exp(H)
    print(f'merge_prob: {merge_prob}')

    result = bool(H > 0 or (merge_prob is not None and merge_prob > torch.rand(1)))
    print(f'result: {result}')

    return result


def log_Hastings_ratio_merge_DPM(
    alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, merge_prob
):
    # use log for overflows
    if N_k_1 == 0:
        lgamma_1 = 0
    else:
        lgamma_1 = lgamma(N_k_1)
    if N_k_2 == 0:
        lgamma_2 = 0
    else:
        lgamma_2 = lgamma(N_k_2)
    # Hastings ratio in log space
    N_k = N_k_1 + N_k_2
    if N_k > 0:
        H = (
            (lgamma(N_k) - (np.log(alpha) + lgamma_1 + lgamma_2))
            + (log_ll_k - (log_ll_k_1 + log_ll_k_2))
        )
    else:
        H = torch.ones(1)

    print(f'Alpha: {alpha}')
    print(f'N_k_1: {N_k_1}')
    print(f'N_k_2: {N_k_2}')
    print(f'Log Likelihood K_1: {log_ll_k_1}')
    print(f'Log Likelihood K_2: {log_ll_k_2}')
    print(f'Log Likelihood K: {log_ll_k}')

    if merge_prob is not None:
        print(f'Merge Probability: {merge_prob.item()}')
    else:
        print('Merge Probability: None')
    
    print(f'Hastings Ratio for Merge: {H.item()}')

    merge_prob = merge_prob or torch.exp(H)
    print("merge_prob",merge_prob)
    print( "bool(H > 0 or (merge_prob is not None and merge_prob > torch.rand(1)))", bool(H > 0 or (merge_prob is not None and merge_prob > torch.rand(1))))
    return bool(H > 0 or (merge_prob is not None and merge_prob > torch.rand(1)))


def split_rule(
    k, codes, logits, logits_sub, covs, covs_sub, cov_const, alpha, split_prob,
    prior=None, ignore_subclusters=False, mus=None, mus_sub=None, n_sub_list=None
):
    """
    Decide whether cluster k should be split. Unlike the original version:
      - We allow partial acceptance of subclusters.
      - Return which subcluster indices are accepted in the final Hastings check.
    """
    threshold = 0  # Minimum number of data points required

    # Number of subclusters for cluster k
    n_sub_k = n_sub_list[k]

    # Get indices of data points assigned to cluster k
    codes_ind = logits.argmax(-1) == k
    codes_k = codes[codes_ind]

    # If fewer than threshold data points in this cluster, do not split
    if len(codes_k) < threshold:
        return [k, False, []]

    # Compute subcluster assignments
    subcluster_offset = sum(n_sub_list[:k])
    if ignore_subclusters:
        # Assign by minimal distance to subcluster means
        mus_sub_k = mus_sub[subcluster_offset: subcluster_offset + n_sub_k]
        codes_k_expanded = codes_k.unsqueeze(1)      # shape [num_points_k, 1, D]
        mus_sub_k_expanded = mus_sub_k.unsqueeze(0)  # shape [1, n_sub_k, D]
        distances = torch.norm(codes_k_expanded - mus_sub_k_expanded, dim=2)
        sub_assignments = distances.argmin(dim=1)
    else:
        # Use logits_sub
        logits_sub_k = logits_sub[codes_ind, subcluster_offset: subcluster_offset + n_sub_k]
        sub_assignments = logits_sub_k.argmax(-1)

    # Now split the codes into subclusters
    codes_k_s_list = []
    for s in range(n_sub_k):
        codes_k_s = codes_k[sub_assignments == s]
        codes_k_s_list.append(codes_k_s)

    # ---------------------------------------------------------------------
    # NEW STEP: Filter subclusters that do not meet the threshold
    # ---------------------------------------------------------------------
    valid_sub_idx = []
    codes_k_s_list_valid = []
    for s in range(n_sub_k):
        if len(codes_k_s_list[s]) > threshold:
            valid_sub_idx.append(s)
            codes_k_s_list_valid.append(codes_k_s_list[s])

    # If fewer than 2 subclusters remain valid, it's not a real split
    if len(codes_k_s_list_valid) < 2:
        # Not enough valid subclusters to proceed
        return [k, False, []]
    # ---------------------------------------------------------------------
    # Now compute log-likelihood only for the valid subclusters
    # ---------------------------------------------------------------------
    # Log marginal likelihood for the main cluster (before split)
    if prior.prior_choice != 'dynamic_data_std':
        log_ll_k = prior.log_marginal_likelihood(codes_k, covs[k])
    else:
        log_ll_k = prior.log_marginal_likelihood_dynamic_psi(codes_k, covs[k], k)

    # Log marginal likelihood for each valid subcluster
    log_ll_k_s_list = []
    for i, codes_k_s in enumerate(codes_k_s_list_valid):
        s_idx = valid_sub_idx[i]
        if prior.prior_choice != 'dynamic_data_std':
            log_ll_k_s = prior.log_marginal_likelihood(
                codes_k_s, covs_sub[subcluster_offset + s_idx]
            )
        else:
            log_ll_k_s = prior.log_marginal_likelihood_dynamic_psi(
                codes_k_s, covs_sub[subcluster_offset + s_idx], k
            )
        log_ll_k_s_list.append(log_ll_k_s)

    # Prior predictive terms before/after split
    qbefore_split = prior.log_mean_projected_data(codes_k,covs[k])
    qafter_split_list = []
    for i, codes_k_s in enumerate(codes_k_s_list_valid):
        s_idx = valid_sub_idx[i]
        mu_s = mus_sub[subcluster_offset + s_idx]
        cov_s = covs_sub[subcluster_offset + s_idx]
        qafter_split_s = prior.log_mean_projected_data(codes_k_s, cov_s)
        qafter_split_list.append(qafter_split_s)
    qafter_split = sum(qafter_split_list)

    # Count of data points in each valid subcluster
    N_s_list = [len(x) for x in codes_k_s_list_valid]

    # Evaluate Hastings ratio
    should_split = log_Hastings_ratio_split(
        alpha, N_s_list, log_ll_k_s_list, log_ll_k, split_prob,
        qbefore_split, qafter_split
    )

    # Return the cluster index, the boolean decision, and the valid subcluster indices
    if should_split:
        return [k, True, valid_sub_idx]
    else:
        return [k, False, []]

def split_rule_previous(
    k, codes, logits, logits_sub, covs, covs_sub, cov_const, alpha, split_prob,
    prior=None, ignore_subclusters=False, mus=None, mus_sub=None, n_sub_list=None
):
    """
    Decide whether cluster k should be split. Unlike the original version:
      - We allow partial acceptance of subclusters.
      - Return which subcluster indices are accepted in the final Hastings check.
    """
    threshold = 0  # Minimum number of data points required

    # Number of subclusters for cluster k
    n_sub_k = n_sub_list[k]

    # Get indices of data points assigned to cluster k
    codes_ind = logits.argmax(-1) == k
    codes_k = codes[codes_ind]

    # If fewer than threshold data points in this cluster, do not split
    if len(codes_k) < threshold:
        return [k, False, []]

    # Compute subcluster assignments
    subcluster_offset = sum(n_sub_list[:k])
    if ignore_subclusters:
        # Assign by minimal distance to subcluster means
        mus_sub_k = mus_sub[subcluster_offset: subcluster_offset + n_sub_k]
        codes_k_expanded = codes_k.unsqueeze(1)      # shape [num_points_k, 1, D]
        mus_sub_k_expanded = mus_sub_k.unsqueeze(0)  # shape [1, n_sub_k, D]
        distances = torch.norm(codes_k_expanded - mus_sub_k_expanded, dim=2)
        sub_assignments = distances.argmin(dim=1)
    else:
        # Use logits_sub
        logits_sub_k = logits_sub[codes_ind, subcluster_offset: subcluster_offset + n_sub_k]
        sub_assignments = logits_sub_k.argmax(-1)

    # Now split the codes into subclusters
    codes_k_s_list = []
    for s in range(n_sub_k):
        codes_k_s = codes_k[sub_assignments == s]
        codes_k_s_list.append(codes_k_s)

    # ---------------------------------------------------------------------
    # NEW STEP: Filter subclusters that do not meet the threshold
    # ---------------------------------------------------------------------
    valid_sub_idx = []
    codes_k_s_list_valid = []
    for s in range(n_sub_k):
        if len(codes_k_s_list[s]) > threshold:
            valid_sub_idx.append(s)
            codes_k_s_list_valid.append(codes_k_s_list[s])

    # If fewer than 2 subclusters remain valid, it's not a real split
    if len(codes_k_s_list_valid) < 2:
        # Not enough valid subclusters to proceed
        return [k, False, []]
    # ---------------------------------------------------------------------
    # Now compute log-likelihood only for the valid subclusters
    # ---------------------------------------------------------------------
    # Log marginal likelihood for the main cluster (before split)
    if prior.prior_choice != 'dynamic_data_std':
        log_ll_k = prior.log_marginal_likelihood(codes_k, covs[k])
    else:
        log_ll_k = prior.log_marginal_likelihood_dynamic_psi(codes_k, covs[k], k)

    # Log marginal likelihood for each valid subcluster
    log_ll_k_s_list = []
    for i, codes_k_s in enumerate(codes_k_s_list_valid):
        s_idx = valid_sub_idx[i]
        if prior.prior_choice != 'dynamic_data_std':
            log_ll_k_s = prior.log_marginal_likelihood(
                codes_k_s, covs_sub[subcluster_offset + s_idx]
            )
        else:
            log_ll_k_s = prior.log_marginal_likelihood_dynamic_psi(
                codes_k_s, covs_sub[subcluster_offset + s_idx], k
            )
        log_ll_k_s_list.append(log_ll_k_s)

    # Prior predictive terms before/after split
    qbefore_split = prior.log_mean_projected_data(codes_k, mus[k], covs[k])
    qafter_split_list = []
    for i, codes_k_s in enumerate(codes_k_s_list_valid):
        s_idx = valid_sub_idx[i]
        mu_s = mus_sub[subcluster_offset + s_idx]
        cov_s = covs_sub[subcluster_offset + s_idx]
        qafter_split_s = prior.log_mean_projected_data(codes_k_s, mu_s, cov_s)
        qafter_split_list.append(qafter_split_s)
    qafter_split = sum(qafter_split_list)

    # Count of data points in each valid subcluster
    N_s_list = [len(x) for x in codes_k_s_list_valid]

    # Evaluate Hastings ratio
    should_split = log_Hastings_ratio_split(
        alpha, N_s_list, log_ll_k_s_list, log_ll_k, split_prob,
        qbefore_split, qafter_split
    )

    # Return the cluster index, the boolean decision, and the valid subcluster indices
    if should_split:
        return [k, True, valid_sub_idx]
    else:
        return [k, False, []]
        
def split_rule_N_subcluster(
    k, codes, logits, logits_sub, covs, covs_sub, cov_const, alpha, split_prob,
    prior=None, ignore_subclusters=False, mus=None, mus_sub=None, n_sub_list=None
):
    """
    Generalized split_rule function for variable number of subclusters per cluster.
    """
    threshold = 5  # Minimum number of data points required

    # Get the number of subclusters for cluster k
    n_sub_k = n_sub_list[k]

    # Get indices of data points assigned to cluster k
    codes_ind = logits.argmax(-1) == k
    codes_k = codes[codes_ind]

    if len(codes_k) < threshold:
        # Not enough data points to consider splitting
        return [k, False]

    # Compute subcluster assignments
    # Compute the starting index for subclusters of cluster k
    subcluster_offset = sum(n_sub_list[:k])

    if ignore_subclusters:
        # Assign points to subclusters based on minimum distance to subcluster means
        mus_sub_k = mus_sub[subcluster_offset: subcluster_offset + n_sub_k]  # Subcluster means for cluster k

        # Compute distances between data points and subcluster means
        codes_k_expanded = codes_k.unsqueeze(1)  # Shape: [num_points_k, 1, D]
        mus_sub_k_expanded = mus_sub_k.unsqueeze(0)  # Shape: [1, n_sub_k, D]
        distances = torch.norm(codes_k_expanded - mus_sub_k_expanded, dim=2)  # Shape: [num_points_k, n_sub_k]

        # Assign each data point to the nearest subcluster
        sub_assignments = distances.argmin(dim=1)  # Shape: [num_points_k]
    else:
        # Get subcluster assignments from logits_sub
        logits_sub_k = logits_sub[codes_ind, subcluster_offset: subcluster_offset + n_sub_k]
        sub_assignments = logits_sub_k.argmax(-1)  # Shape: [num_points_k]

    # Collect data points assigned to each subcluster
    codes_k_s_list = []
    for s in range(n_sub_k):
        codes_k_s = codes_k[sub_assignments == s]
        codes_k_s_list.append(codes_k_s)

    # Check if any subcluster has too few points
    for codes_k_s in codes_k_s_list:
        if len(codes_k_s) <= threshold:
            # Not enough data points in subcluster s to proceed with split
            return [k, False]

    # Compute log marginal likelihoods
    if prior.prior_choice != 'dynamic_data_std':
        log_ll_k = prior.log_marginal_likelihood(codes_k, covs[k])
    else:
        log_ll_k = prior.log_marginal_likelihood_dynamic_psi(codes_k, covs[k], k)

    log_ll_k_s_list = []
    for s, codes_k_s in enumerate(codes_k_s_list):
        if prior.prior_choice != 'dynamic_data_std':
            log_ll_k_s = prior.log_marginal_likelihood(codes_k_s, covs_sub[subcluster_offset + s])
        else:
            log_ll_k_s = prior.log_marginal_likelihood_dynamic_psi(codes_k_s, covs_sub[subcluster_offset + s], k)
        log_ll_k_s_list.append(log_ll_k_s)

    # Get counts of data points in each subcluster
    N_s_list = [len(codes_k_s) for codes_k_s in codes_k_s_list]

    # Compute qbefore_split and qafter_split
    qbefore_split = prior.log_mean_projected_data(codes_k, mus[k], covs[k])

    qafter_split_list = []
    for s, codes_k_s in enumerate(codes_k_s_list):
        mu_s = mus_sub[subcluster_offset + s]
        cov_s = covs_sub[subcluster_offset + s]
        qafter_split_s = prior.log_mean_projected_data(codes_k_s, mu_s, cov_s)
        qafter_split_list.append(qafter_split_s)
    qafter_split = sum(qafter_split_list)  # Sum of log probabilities

    # Compute Hastings ratio and determine if we should split
    should_split = log_Hastings_ratio_split(
        alpha, N_s_list, log_ll_k_s_list, log_ll_k, split_prob, qbefore_split, qafter_split
    )

    return [k, should_split]



def split_rule_2sub(
    k, codes, logits, logits_sub, covs, covs_sub, cov_const, alpha, split_prob, prior=None, ignore_subclusters=False,mus=None,mus_sub=None):
    # look at the points assigned to k
    codes_ind = logits.argmax(-1) == k
    codes_k = codes[codes_ind]

    if len(codes_k) < 5:
        # empty cluster
        return [k, False]

    if ignore_subclusters:
        # comp assignments by min dist
        sub_assignments = comp_subclusters_params_min_dist(codes_k=codes_k, mu_sub_1=mus_sub[2 * k], mu_sub_2=mus_sub[2 * k + 1])
        codes_k_1 = codes_k[sub_assignments == 0]
        codes_k_2 = codes_k[sub_assignments == 1]
    else:
        # subclusters hard assignment
        sub_assignment = logits_sub[codes_ind, :].argmax(-1)
        codes_k_1 = codes_k[sub_assignment == 2 * k]
        codes_k_2 = codes_k[sub_assignment == 2 * k + 1]

    if len(codes_k_1) <= 5 or len(codes_k_2) <= 5:
        # small subclusters
        return [k, False]

    # compute log marginal likelihood
    if prior.prior_choice != 'dynamic_data_std':
      log_ll_k = prior.log_marginal_likelihood(codes_k, covs[k])
      log_ll_k_1 = prior.log_marginal_likelihood(codes_k_1, covs_sub[2 * k])
      log_ll_k_2 = prior.log_marginal_likelihood(codes_k_2, covs_sub[2 * k + 1])
    
    else :
       log_ll_k = prior.log_marginal_likelihood_dynamic_psi(codes_k, covs[k],k)
       log_ll_k_1 = prior.log_marginal_likelihood_dynamic_psi(codes_k_1, covs_sub[2 * k],k)
       log_ll_k_2 = prior.log_marginal_likelihood_dynamic_psi(codes_k_2, covs_sub[2 * k + 1],k)
    
    
    qbefore_split_RandomParamProposalt=prior.log_mean_projected_data(codes_k,  mus[k], covs[k])
    #DeepDPM
    #log_ll_k = prior.log_marginal_likelihood(codes_k, mus[k])
    #log_ll_k_1 = prior.log_marginal_likelihood(codes_k_1, mus_sub[2 * k])
    #log_ll_k_2 = prior.log_marginal_likelihood(codes_k_2, mus_sub[2 * k + 1])

    N_k_1 = len(codes_k_1)
    N_k_2 = len(codes_k_2)

    # use log for overflows
    # Hastings ratio in log space
    return [k, log_Hastings_ratio_split(
        alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, split_prob,qbefore_split_RandomParamProposalt
    )]
    
def split_rule_DPM(
    k, codes, logits, logits_sub, mus, mus_sub, cov_const, alpha, split_prob, prior=None, ignore_subclusters=False
):
    # look at the points assigned to k
    codes_ind = logits.argmax(-1) == k
    codes_k = codes[codes_ind]

    if len(codes_k) < 5:
        # empty cluster
        return [k, False]

    if ignore_subclusters:
        # comp assignments by min dist
        sub_assignments = comp_subclusters_params_min_dist(codes_k=codes_k, mu_sub_1=mus_sub[2 * k], mu_sub_2=mus_sub[2 * k + 1])
        codes_k_1 = codes_k[sub_assignments == 0]
        codes_k_2 = codes_k[sub_assignments == 1]
    else:
        # subclusters hard assignment
        sub_assignment = logits_sub[codes_ind, :].argmax(-1)
        codes_k_1 = codes_k[sub_assignment == 2 * k]
        codes_k_2 = codes_k[sub_assignment == 2 * k + 1]

    if len(codes_k_1) <= 5 or len(codes_k_2) <= 5:
        # small subclusters
        return [k, False]

    # compute log marginal likelihood
    log_ll_k = prior.log_marginal_likelihood(codes_k, mus[k])
    log_ll_k_1 = prior.log_marginal_likelihood(codes_k_1, mus_sub[2 * k])
    log_ll_k_2 = prior.log_marginal_likelihood(codes_k_2, mus_sub[2 * k + 1])
    
    #DeepDPM
    #log_ll_k = prior.log_marginal_likelihood(codes_k, mus[k])
    #log_ll_k_1 = prior.log_marginal_likelihood(codes_k_1, mus_sub[2 * k])
    #log_ll_k_2 = prior.log_marginal_likelihood(codes_k_2, mus_sub[2 * k + 1])

    N_k_1 = len(codes_k_1)
    N_k_2 = len(codes_k_2)

    # use log for overflows
    # Hastings ratio in log space
    return [k, log_Hastings_ratio_split(
        alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, split_prob
    )]


def compute_split_log_marginal_ll():
    pass


def compute_split_log_ll(
    mu, mus_sub_1, mus_sub_2, cov_const, codes_k, codes_k_1, codes_k_2
):
    D = len(mu)
    dist_k = torch.distributions.multivariate_normal.MultivariateNormal(
        mu, torch.eye(D) * cov_const
    )
    dist_k_1 = torch.distributions.multivariate_normal.MultivariateNormal(
        mus_sub_1, torch.eye(D) * cov_const
    )
    dist_k_2 = torch.distributions.multivariate_normal.MultivariateNormal(
        mus_sub_2, torch.eye(D) * cov_const
    )

    log_ll_k = dist_k.log_prob(codes_k).sum()
    log_ll_k_1 = (dist_k_1.log_prob(codes_k_1)).sum()
    log_ll_k_2 = (dist_k_2.log_prob(codes_k_2)).sum()

    return log_ll_k, log_ll_k_1, log_ll_k_2


def split_step(
    K, codes, logits, logits_sub, covs, covs_sub, cov_const,
    n_sub_list, alpha, split_prob, prior=None, ignore_subclusters=False,
    mus=None, mus_sub=None
):
    """
    Attempt splits for each of K clusters, returning:
      1) A Boolean tensor [K] with split decisions (True/False).
      2) A list of lists 'accepted_subclusters', where accepted_subclusters[k]
         is the list of subcluster indices that were actually used for cluster k.
    """
    # List of decisions: each entry [cluster_index, bool_decision]
    split_decisions_raw = []
    # List of accepted subclusters for each cluster k
    accepted_subclusters = [[] for _ in range(K)]

    # Loop over each cluster and call split_rule
    for k in range(K):
        k_out, decision, valid_subs = split_rule(
            k, codes, logits, logits_sub, covs, covs_sub, cov_const, alpha, split_prob,
            prior=prior, ignore_subclusters=ignore_subclusters,
            mus=mus, mus_sub=mus_sub, n_sub_list=n_sub_list
        )
        split_decisions_raw.append([k_out, decision])
        # If split is accepted, record which subclusters got used
        if decision:
            accepted_subclusters[k_out] = valid_subs
        else:
            accepted_subclusters[k_out] = []

    # Sort decisions into a Boolean tensor of shape (K,)
    temp = torch.empty(K, dtype=bool)
    if isinstance(split_decisions_raw[0][1], np.bool_):
        # Convert np.bool_ -> torch.bool
        for i in range(K):
            temp[split_decisions_raw[i][0]] = torch.tensor(split_decisions_raw[i][1], dtype=torch.bool)
    else:
        for i in range(K):
            temp[split_decisions_raw[i][0]] = split_decisions_raw[i][1]
    split_decisions = temp

    print('split_decisions (bool tensor):', split_decisions)
    print('accepted_subclusters:', accepted_subclusters)
    
    # Return both the boolean decisions and the accepted subcluster indices
    return split_decisions, accepted_subclusters
    
def split_step_N_subcluster(
    K, codes, logits, logits_sub, covs, covs_sub, cov_const,n_sub_list, alpha, split_prob, prior=None,ignore_subclusters=False,mus=None,mus_sub=None):
    # Get split decision for all the clusters in parallel
    # from joblib import Parallel, delayed
    # split_decisions = Parallel(n_jobs=2)(delayed(split_rule)(
    #     k, codes, logits, logits_sub, mus, mus_sub, cov_const, alpha, split_prob, prior=prior, ignore_subclusters=ignore_subclusters) for k in range(K))

    # returns for each cluster a list [k, True/False]
    split_decisions = []
    for k in range(K):
         split_decisions.append(split_rule(
    k, codes, logits, logits_sub, covs, covs_sub, cov_const, alpha, split_prob,
    prior=prior, ignore_subclusters=ignore_subclusters, mus=mus, mus_sub=mus_sub, n_sub_list=n_sub_list))

    # sort list
    temp = torch.empty(K, dtype=bool)
    #print('split_decisions :',split_decisions)
    print('split_decisions type :',isinstance(split_decisions[0][1],np.bool_))
    if isinstance(split_decisions[0][1],np.bool_):
      for i in range(K):
        temp[split_decisions[i][0]] = torch.Tensor(split_decisions[i][1],dtype=torch.bool)
    else :
      for i in range(K):
        temp[split_decisions[i][0]] = split_decisions[i][1]
    split_decisions = temp
    print('split_decisions :',split_decisions)
    return split_decisions


def update_clusters_prior_split(codes, logits, prior, split_decisions, mus_sub, logits_sub, n_sub_list):
    """
    Updates the priors for clusters following a split operation, handling variable numbers of subclusters per cluster.

    Args:
        codes (torch.Tensor): The original data points.
        logits (torch.Tensor): The logits corresponding to the original clusters.
        prior: The current prior object.
        split_decisions (torch.Tensor): A boolean tensor indicating which clusters were split.
        mus_sub (torch.Tensor): The subclusters' mus after the split.
        logits_sub (torch.Tensor): The logits for the subclusters.
        n_sub_list (List[int]): List of number of subclusters per cluster.

    Returns:
        Updated prior object.
    """
    # Ensure split_decisions is a boolean tensor
    split_decisions = split_decisions.bool()

    # Indices of clusters not being split
    indices_not_split = (~split_decisions).nonzero(as_tuple=False).squeeze()
    
    #indices_split = split_decisions.nonzero(as_tuple=False).squeeze()

    # Step 1: Extract priors for non-split clusters
    prior_not_split = prior.mus_covs_prior.niw_psi_clusters[indices_not_split]
    if prior_not_split.dim() == 2:
        prior_not_split = prior_not_split.unsqueeze(0)

    prior_new_list = [prior_not_split]

    # Total number of subclusters for clusters not being split
    n_sub_list_not_split = [n_sub_list[k] for k in range(len(n_sub_list)) if not split_decisions[k]]
    total_subclusters_not_split = sum(n_sub_list_not_split)

    # Offset for subclusters of clusters being split in logits_sub
    subcluster_offset_split = 0

    for k in range(len(split_decisions)):
        num_subclusters_k = n_sub_list[k]
        if split_decisions[k]:
            # Get data points assigned to cluster k
            indices_k = logits.argmax(dim=1) == k
            codes_k = codes[indices_k, :]

            # Get responsibilities for subclusters of cluster k
            start_idx = total_subclusters_not_split + subcluster_offset_split
            end_idx = start_idx + num_subclusters_k
            r_sub = logits_sub[indices_k, start_idx:end_idx]

            # Compute priors for each subcluster of cluster k
            for k_sub in range(num_subclusters_k):
                z_sub = r_sub[:, k_sub]  # Responsibilities of data points to subcluster k_sub

                if z_sub.sum() > 0:
                    # Calculate Karcher mean for subcluster
                    mus_sub_k = KarcherMean(z_sub, codes_k)
                    # Perform log mapping of data points onto the tangent space at the Karcher mean
                    log_map_k = Log_mapping(codes_k, mus_sub_k)
                    # Compute standard deviation in the log-mapped space
                    std_k = log_map_k.std(dim=0)
                    # Compute the prior covariance matrix
                    prior_k = (torch.diag(std_k) * prior.get_prior_sigma_scale()).unsqueeze(0)
                else:
                    # If no data points assigned, set prior as zeros
                    prior_k = torch.zeros((codes_k.size(1), codes_k.size(1)), dtype=torch.double).unsqueeze(0)

                # Append the prior for the subcluster
                prior_new_list.append(prior_k)

            subcluster_offset_split += num_subclusters_k
        else:
            # Cluster not being split; nothing to do
            pass

    # Step 3: Concatenate the priors for split clusters with non-split cluster priors
    prior_new = torch.cat(prior_new_list, dim=0)
    prior.mus_covs_prior.niw_psi_clusters = prior_new

    return prior

def update_clusters_prior_split_2sub(codes, logits, prior, split_decisions, mus_ind_to_split, mus_sub, logits_sub):
    """
    Updates the priors for clusters following a split operation.

    Args:
        codes (torch.tensor): The original data points.
        logits (torch.tensor): The logits corresponding to the original clusters.
        prior: The current prior object.
        split_decisions (list): A boolean list of length(mus) with True where mus_ind was split.
        mus_ind_to_split (list): A list of the indices of mus that were chosen to be split.
        mus_sub (torch.tensor): The subclusters' mus after the split.
        logits_sub (torch.tensor): The logits for the subclusters.

    Returns:
        torch.tensor: The updated priors for all clusters.
    """
    # Step 1: Extract priors for non-split clusters
    prior_new = prior.mus_covs_prior.niw_psi_clusters[torch.logical_not(split_decisions)]
    if prior_new.dim() == 2: 
        prior_new = prior_new.unsqueeze(0)
    
    # Step 2: Prepare a list to accumulate the new priors for split clusters
    priors_to_add = []
    
    for k in mus_ind_to_split:
        # Select the data points corresponding to the original cluster `k`
        indices_k = logits.argmax(-1) == k
        codes_k = codes[indices_k, :]
        
        # Compute responsibilities for each subcluster of `k`
        r_sub = logits_sub[indices_k, 2 * k: 2 * k + 2]
        
        # List to store priors for subclusters of `k`
        priors_sub_k = []
        
        for k_sub in range(2):  # Assuming each cluster is split into 2 subclusters
            z_sub = r_sub[:, k_sub]  # Responsibility of each data point to subcluster `k_sub`
            
            if z_sub.sum() > 0:  # Check to ensure there are data points assigned
                # Calculate Karcher mean for subcluster
                mus_sub_k = KarcherMean(z_sub, codes_k)
                # Perform log mapping of data points onto the tangent space at the Karcher mean
                log_map_k = Log_mapping(codes_k, mus_sub_k)
                # Compute standard deviation in the log-mapped space
                std_k = log_map_k.std(axis=0)
                # Compute the prior covariance matrix
                prior_k = (torch.diag(std_k) * prior.get_prior_sigma_scale()).unsqueeze(0)

            else:
                # If no data points assigned, set prior as a zero matrix of appropriate size
                prior_k = (torch.zeros((codes_k.size(1), codes_k.size(1))).double()).unsqueeze(0)
                
            # Append the prior for the subcluster
            priors_sub_k.append(prior_k.double())

        # Add the priors of the subclusters to the list of new priors
        priors_to_add.extend(priors_sub_k)
    
    # Step 3: Concatenate the priors for split clusters with non-split cluster priors
    if len(priors_to_add) > 0:
        prior_new = torch.cat([prior_new] + priors_to_add)
    
    prior.mus_covs_prior.niw_psi_clusters=prior_new
    
    return prior

def update_clusters_params_split(
    mus,
    covs,
    pi,
    split_decisions,
    mus_sub,
    covs_sub,
    pi_sub,
    prior,
    codes,
    logits,
    logits_sub,
    n_sub_list,
    cluster_labels,
    prior_labels,
    accepted_subclusters=None  # <-- NEW
):
    """
    Update parameters for main clusters after a split. 
    If a cluster is not split, it stays as 1 new cluster. 
    If it is split, we create a new main cluster for each accepted subcluster.

    Returns:
    --------
    (
        mus_new, covs_new, pi_new, prior_new,
        updated_cluster_labels, updated_prior_labels,
        origin_cardinality  # <-- new dictionary
    )
    """
    # 1) Possibly update prior (unchanged logic):
    if prior.prior_choice == 'dynamic_data_std':
        prior_new = update_clusters_prior_split(
            codes, logits, prior, split_decisions, mus_sub, logits_sub, n_sub_list
        )
    else:
        prior_new = prior

    split_decisions = split_decisions.bool()
    indices_not_split = (~split_decisions).nonzero(as_tuple=False).squeeze()
    if indices_not_split.ndim == 0:
        indices_not_split = indices_not_split.unsqueeze(0)

    # Keep clusters not being split
    mus_new  = mus[indices_not_split]
    covs_new = covs[indices_not_split]
    pi_new   = pi[indices_not_split]

    updated_cluster_labels = {}
    updated_prior_labels   = {}

    # -- NEW: We'll build this dictionary: { new_cluster_idx: int }
    #    specifying how many subclusters formed this new main cluster
    origin_cardinality = {}

    new_cluster_idx = 0

    # ---------------------------------------------------------
    # (A) Handle clusters NOT being split
    # ---------------------------------------------------------
    for old_idx in indices_not_split.tolist():
        updated_cluster_labels[new_cluster_idx] = cluster_labels[old_idx]
        updated_prior_labels[new_cluster_idx]   = prior_labels[old_idx]

        # Not split => 1 main cluster => let's define origin_cardinality=1 
        # or you could default to your n_merge
        origin_cardinality[new_cluster_idx] = 1

        new_cluster_idx += 1

    # ---------------------------------------------------------
    # (B) Handle clusters that ARE being split
    # ---------------------------------------------------------
    split_indices = split_decisions.nonzero(as_tuple=False).squeeze()
    if split_indices.ndim == 0:
        split_indices = split_indices.unsqueeze(0)
    split_indices = split_indices.tolist()

    mus_to_add  = []
    covs_to_add = []
    pis_to_add  = []

    for k in split_indices:
        # Indices for subclusters that belong to old cluster k
        start_idx = sum(n_sub_list[:k])
        end_idx   = start_idx + n_sub_list[k]

        mus_sub_k = mus_sub[start_idx:end_idx]
        covs_sub_k = covs_sub[start_idx:end_idx]
        pi_sub_k   = pi_sub[start_idx:end_idx]

        old_label = cluster_labels[k]

        # partial acceptance
        if accepted_subclusters is not None:
            sub_idx_list = accepted_subclusters[k]  # e.g. [0, 2]
        else:
            sub_idx_list = range(n_sub_list[k])

        # number of accepted subclusters
        L = len(sub_idx_list)

        for i in sub_idx_list:
            mus_to_add.append(mus_sub_k[i].unsqueeze(0))
            covs_to_add.append(covs_sub_k[i].unsqueeze(0))
            pis_to_add.append(pi_sub_k[i].unsqueeze(0))

            new_label = f"{old_label}{i+1}"
            updated_cluster_labels[new_cluster_idx] = new_label
            updated_prior_labels[new_cluster_idx]   = new_label

            # This new main cluster originated from a cluster that had L accepted subclusters
            origin_cardinality[new_cluster_idx] = L

            new_cluster_idx += 1

    # ---------------------------------------------------------
    # (C) Concatenate new rows if we created them
    # ---------------------------------------------------------
    if mus_to_add:
        mus_to_add  = torch.cat(mus_to_add,  dim=0)
        covs_to_add = torch.cat(covs_to_add, dim=0)
        pis_to_add  = torch.cat(pis_to_add,  dim=0)

        mus_new  = torch.cat([mus_new,  mus_to_add],  dim=0)
        covs_new = torch.cat([covs_new, covs_to_add], dim=0)
        pi_new   = torch.cat([pi_new,   pis_to_add],  dim=0)

    # Re-normalize pi
    pi_new = pi_new / pi_new.sum()

    # Return everything + origin_cardinality
    return (
        mus_new,
        covs_new,
        pi_new,
        prior_new,
        updated_cluster_labels,
        updated_prior_labels,
        origin_cardinality  # <-- new
    )

def update_clusters_params_split_N_subcluster(
    mus,
    covs,
    pi,
    split_decisions,
    mus_sub,
    covs_sub,
    pi_sub,
    prior,
    codes,
    logits,
    logits_sub,
    n_sub_list,
    cluster_labels,
    prior_labels
):
    # Update prior if necessary
    if prior.prior_choice == 'dynamic_data_std':
        prior_new = update_clusters_prior_split(
            codes, logits, prior, split_decisions, mus_sub, logits_sub, n_sub_list
        )
    else:
        prior_new = prior

    # Clusters not being split
    split_decisions = split_decisions.bool()
    indices_not_split = (~split_decisions).nonzero(as_tuple=False).squeeze()
    if indices_not_split.ndim ==0:
       indices_not_split=torch.tensor([indices_not_split.item()])
    
    mus_new = mus[indices_not_split]
    covs_new = covs[indices_not_split]
    pi_new = pi[indices_not_split]

    updated_cluster_labels = {}
    updated_prior_labels = {}
    new_cluster_idx = 0

    # Update labels for clusters not split
    for old_idx in indices_not_split.tolist():
        updated_cluster_labels[new_cluster_idx] = cluster_labels[old_idx]
        updated_prior_labels[new_cluster_idx] = prior_labels[old_idx]
        new_cluster_idx += 1

    # Collect parameters of clusters to add (from subclusters of clusters being split)
    mus_to_add = []
    covs_to_add = []
    pis_to_add = []

    # Update labels for clusters being split
    #split_indices = split_decisions.nonzero(as_tuple=False).squeeze().tolist()
    split_indices = split_decisions.nonzero(as_tuple=False).squeeze()
    if split_indices.ndim ==0:
      split_indices=torch.tensor([split_indices.item()])
    split_indices=split_indices.tolist()
    for k in split_indices:
        num_subclusters_k = n_sub_list[k]
        # Indices in mus_sub, covs_sub, pi_sub
        start_idx = sum(n_sub_list[:k])
        end_idx = start_idx + num_subclusters_k
        mus_sub_k = mus_sub[start_idx:end_idx]
        covs_sub_k = covs_sub[start_idx:end_idx]
        pi_sub_k = pi_sub[start_idx:end_idx]
        # Append to the lists
        mus_to_add.append(mus_sub_k)
        covs_to_add.append(covs_sub_k)
        pis_to_add.append(pi_sub_k)
        # Update labels
        old_label = cluster_labels[k]
        for i in range(num_subclusters_k):
            new_label = old_label + str(i + 1)
            updated_cluster_labels[new_cluster_idx] = new_label
            updated_prior_labels[new_cluster_idx] = new_label
            new_cluster_idx += 1

    # Concatenate the new clusters
    if mus_to_add:
        mus_to_add = torch.cat(mus_to_add, dim=0)
        covs_to_add = torch.cat(covs_to_add, dim=0)
        pis_to_add = torch.cat(pis_to_add, dim=0)
        mus_new = torch.cat([mus_new, mus_to_add], dim=0)
        covs_new = torch.cat([covs_new, covs_to_add], dim=0)
        pi_new = torch.cat([pi_new, pis_to_add], dim=0)

    # Normalize pi_new to sum to 1
    pi_new = pi_new / pi_new.sum() # CA NETAIT PAS LA  DANS LA VERSION 2SUB

    return mus_new, covs_new, pi_new, prior_new, updated_cluster_labels, updated_prior_labels



def update_clusters_params_split_2sub(
    mus, covs, pi, mus_ind_to_split, split_decisions, mus_sub, covs_sub, pi_sub,prior,codes,logits,logits_sub
):
    """This function is used to compute the new model parameters following a split

    Args:
        mus ([torch.tensor]): The mus before the split
        covs ([torch.tensor]): The covs before the split
        pi ([torch.tensor]): The pis before the split
        mus_ind_to_split ([list]): A list of the mus that were chosen to be split
        split_decisions ([list]): A boolean list of len(mus) with True where mus_ind was split
        mus_sub ([type]): The subclusters' mus before the split

    Returns:
        mus_new ([torch.tensor]), covs_new ([torch.tensor]), pi_new ([torch.tensor]): The new parameters
    """
    if prior.prior_choice == 'dynamic_data_std':
      prior_new=update_clusters_prior_split(codes, logits, prior, split_decisions, mus_ind_to_split, mus_sub, logits_sub)
    else :
      prior_new=prior
    mus_new = mus[torch.logical_not(split_decisions)]
    covs_new = covs[torch.logical_not(split_decisions)]
    pi_new = pi[torch.logical_not(split_decisions)]
 
    mus_to_add, covs_to_add, pis_to_add = [], [], []
    for k in mus_ind_to_split:
        mus_to_add.extend([mus_sub[2 * k], mus_sub[2 * k + 1]])
        covs_to_add.extend([covs_sub[2 * k], covs_sub[2 * k + 1]])
        pis_to_add.extend([pi_sub[2 * k], pi_sub[2 * k + 1]])

    mus_new = torch.cat([mus_new, torch.cat(mus_to_add)])
    covs_new = torch.cat([covs_new, torch.cat(covs_to_add)])
    pi_new = torch.cat([pi_new, torch.cat(pis_to_add)])
    
    return mus_new, covs_new, pi_new , prior_new

def update_subclusters_params_split_original(
    mus_sub,
    covs_sub,
    pi_sub,
    split_decisions,
    codes,
    logits,
    logits_sub,
    n_sub_list,
    n_sub,
    how_to_init_mu_sub,
    prior,
    use_priors=True
):
    """
    Generalized update_subclusters_params_split function to handle variable number of subclusters per cluster,
    ensuring that subclusters are added at the end and in the order of the clusters that have been split.

    Parameters:
    - mus_sub: Tensor of subcluster means.
    - covs_sub: Tensor of subcluster covariances.
    - pi_sub: Tensor of subcluster mixing coefficients.
    - split_decisions: Tensor of booleans indicating which clusters to split.
    - codes: Tensor of data points.
    - logits: Tensor of main cluster logits.
    - logits_sub: Tensor of subcluster logits.
    - n_sub_list: List containing the number of subclusters per cluster.
    - how_to_init_mu_sub: Method to initialize subcluster means ('umap' only).
    - prior: Prior object.
    - use_priors: Whether to use prior distributions in calculations.

    Returns:
    - mus_sub_new: Updated tensor of subcluster means.
    - covs_sub_new: Updated tensor of subcluster covariances.
    - pi_sub_new: Updated tensor of subcluster mixing coefficients.
    - n_sub_list_new: Updated list of subcluster counts per cluster.
    """
    mus_sub_new = []
    covs_sub_new = []
    pi_sub_new = []
    n_sub_list_new = []

    subcluster_offset = 0
    split_indices = split_decisions.nonzero(as_tuple=False).squeeze().tolist()
    if isinstance(split_indices, int):
        split_indices = [split_indices]

    # First, process clusters not being split
    for k in range(len(split_decisions)):
        num_subclusters_k = n_sub_list[k]
        if not split_decisions[k]:
            # Cluster not split; keep its subclusters
            start_idx = subcluster_offset
            end_idx = subcluster_offset + num_subclusters_k
            mus_sub_k = mus_sub[start_idx:end_idx]
            covs_sub_k = covs_sub[start_idx:end_idx]
            pi_sub_k = pi_sub[start_idx:end_idx]

            mus_sub_new.extend(mus_sub_k)
            covs_sub_new.extend(covs_sub_k)
            pi_sub_new.extend(pi_sub_k)
            n_sub_list_new.append(num_subclusters_k)
        subcluster_offset += num_subclusters_k

    # Now, process clusters being split and add their new subclusters at the end
    for k in split_indices:
        num_subclusters_k = n_sub_list[k]
        start_idx = sum(n_sub_list[:k])
        end_idx = start_idx + num_subclusters_k

        for i in range(num_subclusters_k):
            k_sub = start_idx + i
            # Determine the maximum number of subclusters for the new cluster
            #n_sub_max = n_sub_list[k]  # Adjust as needed

            # Call _create_subclusters for k_sub
            new_mus_sub, new_covs_sub, new_pi_sub, best_n_subclusters = _create_subclusters(
                k_sub=k_sub,
                codes=codes,
                logits=logits,
                logits_sub=logits_sub,
                mus_sub=mus_sub,
                pi_sub=pi_sub,
                n_sub_max=n_sub,
                how_to_init_mu_sub=how_to_init_mu_sub,
                prior=prior,
                n_sub_list=n_sub_list,
                use_priors=use_priors
            )
            mus_sub_new.extend(new_mus_sub)
            covs_sub_new.extend(new_covs_sub)
            pi_sub_new.extend(new_pi_sub)
            n_sub_list_new.append(best_n_subclusters)

    # Convert lists to tensors
    mus_sub_new = torch.stack(mus_sub_new)
    covs_sub_new = torch.stack(covs_sub_new)
    pi_sub_new = torch.stack(pi_sub_new)

    return mus_sub_new, covs_sub_new, pi_sub_new, n_sub_list_new

def update_subclusters_params_split(
    mus_sub,
    covs_sub,
    pi_sub,
    split_decisions,
    codes,
    logits,
    logits_sub,
    n_sub_list,
    n_sub,
    how_to_init_mu_sub,
    prior,
    use_priors=True,
    subcluster_labels=None,
    updated_cluster_labels=None,
    accepted_subclusters=None  # <-- NEW
):
    mus_sub_new = []
    covs_sub_new = []
    pi_sub_new = []
    n_sub_list_new = []
    updated_subcluster_labels_new = {}

    subcluster_offset = 0
    split_indices = split_decisions.nonzero(as_tuple=False).squeeze()
    if split_indices.ndim ==0:
        split_indices = split_indices.unsqueeze(0)
    split_indices = split_indices.tolist()

    # ---------------------------------------------
    # First, keep clusters not being split as-is
    # ---------------------------------------------
    for k in range(len(split_decisions)):
        num_subclusters_k = n_sub_list[k]
        if not split_decisions[k]:
            start_idx = subcluster_offset
            end_idx = subcluster_offset + num_subclusters_k
            mus_sub_k = mus_sub[start_idx:end_idx]
            covs_sub_k = covs_sub[start_idx:end_idx]
            pi_sub_k = pi_sub[start_idx:end_idx]

            mus_sub_new.extend(mus_sub_k)
            covs_sub_new.extend(covs_sub_k)
            pi_sub_new.extend(pi_sub_k)
            n_sub_list_new.append(num_subclusters_k)

            # Label update
            if updated_cluster_labels is not None and subcluster_labels is not None:
                new_cluster_idx = len(n_sub_list_new) - 1
                if k in subcluster_labels:
                    updated_subcluster_labels_new[new_cluster_idx] = subcluster_labels[k].copy()
                else:
                    updated_subcluster_labels_new[new_cluster_idx] = {}
        subcluster_offset += num_subclusters_k

    # ---------------------------------------------
    # Then, for each cluster that *is* being split
    # we create brand-new subclusters (or do your
    # "ghost indexing") if that is your model logic.
    # ---------------------------------------------
    for k in split_indices:
        num_subclusters_k = n_sub_list[k]
        start_idx = sum(n_sub_list[:k])
        end_idx = start_idx + num_subclusters_k

        # Instead of enumerating all i in range(num_subclusters_k), 
        # use accepted_subclusters if you only want to handle certain subclusters
        if accepted_subclusters is not None:
            sub_idx_list = accepted_subclusters[k]
        else:
            sub_idx_list = range(num_subclusters_k)

        # Now, for each accepted subcluster i, we call _create_subclusters, or
        # do whatever logic you want for splitting into further sub-subclusters
        for i in sub_idx_list:
            k_sub = start_idx + i
            # E.g., your function that initializes new sub-subclusters from i
            (new_mus_sub, new_covs_sub, new_pi_sub, best_n_subclusters) = _create_subclusters(
                k_sub=k_sub,
                codes=codes,
                logits=logits,
                logits_sub=logits_sub,
                mus_sub=mus_sub,
                pi_sub=pi_sub,
                n_sub_max=n_sub,
                how_to_init_mu_sub=how_to_init_mu_sub,
                prior=prior,
                n_sub_list=n_sub_list,
                use_priors=use_priors
            )

            mus_sub_new.extend(new_mus_sub)
            covs_sub_new.extend(new_covs_sub)
            pi_sub_new.extend(new_pi_sub)
            n_sub_list_new.append(best_n_subclusters)

            # Labeling logic
            new_cluster_idx = len(n_sub_list_new) - 1
            original_label = updated_cluster_labels.get(new_cluster_idx, f"Cluster{new_cluster_idx}")
            updated_subcluster_labels_new[new_cluster_idx] = {}
            for s in range(best_n_subclusters):
                new_label = f"{original_label}{s + 1}"
                updated_subcluster_labels_new[new_cluster_idx][s] = new_label

    # Finally, convert to tensors
    if mus_sub_new:
        mus_sub_new = torch.stack(mus_sub_new, dim=0)
        covs_sub_new = torch.stack(covs_sub_new, dim=0)
        pi_sub_new = torch.stack(pi_sub_new, dim=0)
    else:
        mus_sub_new = torch.empty((0, mus_sub.size(1)), device=mus_sub.device)
        covs_sub_new = torch.empty((0, covs_sub.size(1)), device=covs_sub.device)
        pi_sub_new = torch.empty((0,), device=pi_sub.device)

    return mus_sub_new, covs_sub_new, pi_sub_new, n_sub_list_new, updated_subcluster_labels_new


def update_subclusters_params_split_N_subcluster(
    mus_sub,
    covs_sub,
    pi_sub,
    split_decisions,
    codes,
    logits,
    logits_sub,
    n_sub_list,
    n_sub,
    how_to_init_mu_sub,
    prior,
    use_priors=True,
    subcluster_labels=None,
    updated_cluster_labels=None
):
    """
    Updated function to handle ghost indexing by tracking cluster and subcluster labels.
    Subcluster labels follow the format 'A11', 'A12', etc., without underscores.

    Parameters:
    - mus_sub: Tensor of subcluster means.
    - covs_sub: Tensor of subcluster covariances.
    - pi_sub: Tensor of subcluster mixing coefficients.
    - split_decisions: Tensor of booleans indicating which clusters to split.
    - codes: Tensor of data points.
    - logits: Tensor of main cluster logits.
    - logits_sub: Tensor of subcluster logits.
    - n_sub_list: List containing the number of subclusters per cluster.
    - n_sub: Maximum number of subclusters to split into.
    - how_to_init_mu_sub: Method to initialize subcluster means ('umap' only).
    - prior: Prior object.
    - use_priors: Whether to use prior distributions in calculations.
    - subcluster_labels: Dictionary mapping cluster indices to their subcluster label dictionaries.
    - updated_cluster_labels: Dictionary mapping updated cluster indices to labels.

    Returns:
    - mus_sub_new: Updated tensor of subcluster means.
    - covs_sub_new: Updated tensor of subcluster covariances.
    - pi_sub_new: Updated tensor of subcluster mixing coefficients.
    - n_sub_list_new: Updated list of subcluster counts per cluster.
    - updated_subcluster_labels_new: Updated dictionary of subcluster labels.
    """
    mus_sub_new = []
    covs_sub_new = []
    pi_sub_new = []
    n_sub_list_new = []

    # Initialize updated subcluster labels
    updated_subcluster_labels_new = {}

    subcluster_offset = 0
    #split_indices = split_decisions.nonzero(as_tuple=False).squeeze().tolist()
    split_indices = split_decisions.nonzero(as_tuple=False).squeeze()
    if split_indices.ndim ==0:
      split_indices=torch.tensor([split_indices.item()])
    split_indices=split_indices.tolist()
    if isinstance(split_indices, int):
        split_indices = [split_indices]

    # First, process clusters not being split
    for k in range(len(split_decisions)):
        num_subclusters_k = n_sub_list[k]
        if not split_decisions[k]:
            # Cluster not split; keep its subclusters
            start_idx = subcluster_offset
            end_idx = subcluster_offset + num_subclusters_k
            mus_sub_k = mus_sub[start_idx:end_idx]
            covs_sub_k = covs_sub[start_idx:end_idx]
            pi_sub_k = pi_sub[start_idx:end_idx]

            mus_sub_new.extend(mus_sub_k)
            covs_sub_new.extend(covs_sub_k)
            pi_sub_new.extend(pi_sub_k)
            n_sub_list_new.append(num_subclusters_k)

            # Update subcluster labels
            if updated_cluster_labels is not None and subcluster_labels is not None:
                # The new cluster index is len(n_sub_list_new) -1
                new_cluster_idx = len(n_sub_list_new) - 1
                # Assign existing subcluster labels
                # Ensure that the original cluster has corresponding subcluster labels
                if k in subcluster_labels:
                    # Copy existing subcluster labels to the new cluster index
                    updated_subcluster_labels_new[new_cluster_idx] = subcluster_labels[k].copy()
                else:
                    # If no labels exist, initialize empty dictionary
                    updated_subcluster_labels_new[new_cluster_idx] = {}
        subcluster_offset += num_subclusters_k

    # Now, process clusters being split and add their new subclusters at the end
    for k in split_indices:
        num_subclusters_k = n_sub_list[k]
        start_idx = sum(n_sub_list[:k])
        end_idx = start_idx + num_subclusters_k

        for i in range(num_subclusters_k):
            k_sub = start_idx + i
            # Call _create_subclusters for k_sub
            new_mus_sub, new_covs_sub, new_pi_sub, best_n_subclusters = _create_subclusters(
                k_sub=k_sub,
                codes=codes,
                logits=logits,
                logits_sub=logits_sub,
                mus_sub=mus_sub,
                pi_sub=pi_sub,
                n_sub_max=n_sub,
                how_to_init_mu_sub=how_to_init_mu_sub,
                prior=prior,
                n_sub_list=n_sub_list,
                use_priors=use_priors
            )
            mus_sub_new.extend(new_mus_sub)
            covs_sub_new.extend(new_covs_sub)
            pi_sub_new.extend(new_pi_sub)
            n_sub_list_new.append(best_n_subclusters)

            # Update subcluster labels
            if updated_cluster_labels is not None and subcluster_labels is not None:
                # The new cluster index is len(n_sub_list_new) -1
                new_cluster_idx = len(n_sub_list_new) - 1
                original_label = updated_cluster_labels.get(new_cluster_idx, f"Cluster{new_cluster_idx}")
                # Assign new labels for the subclusters without underscores
                # For example, if original_label is 'A1', subclusters will be 'A11', 'A12', etc.
                updated_subcluster_labels_new[new_cluster_idx] = {}
                for s in range(best_n_subclusters):
                    # Concatenate without underscore
                    new_label = f"{original_label}{s + 1}"
                    updated_subcluster_labels_new[new_cluster_idx][s] = new_label

    # Convert lists to tensors
    if mus_sub_new:
        mus_sub_new = torch.stack(mus_sub_new)
        covs_sub_new = torch.stack(covs_sub_new)
        pi_sub_new = torch.stack(pi_sub_new)
    else:
        # Handle the case where no subclusters are present
        mus_sub_new = torch.empty((0, mus_sub.size(1)), device=mus_sub.device)
        covs_sub_new = torch.empty((0, covs_sub.size(1)), device=covs_sub.device)
        pi_sub_new = torch.empty((0,), device=pi_sub.device)

    return mus_sub_new, covs_sub_new, pi_sub_new, n_sub_list_new, updated_subcluster_labels_new




def update_subclusters_params_split_2sub(
    mus_sub,
    covs_sub,
    pi_sub,
    mus_ind_to_split,
    split_decisions,
    codes,
    logits,
    logits_sub,
    n_sub,
    how_to_init_mu_sub,
    prior,
    use_priors=True
):
    mus_sub_new = mus_sub[
        torch.logical_not(split_decisions).repeat_interleave(2)
    ]
    covs_sub_new = covs_sub[
        torch.logical_not(split_decisions).repeat_interleave(2)
    ]
    pi_sub_new = pi_sub[
        torch.logical_not(split_decisions).repeat_interleave(2)
    ]
    mus_sub_to_add, covs_sub_to_add, pis_sub_to_add = [], [], []
    for k in mus_ind_to_split:
        (
            new_mus_sub_1,
            new_covs_sub_1,
            new_pis_1,
        ) = _create_subclusters(
            k_sub=2 * k,
            codes=codes,
            logits=logits,
            logits_sub=logits_sub,
            mus_sub=mus_sub,
            pi_sub=pi_sub,
            n_sub=n_sub,
            how_to_init_mu_sub=how_to_init_mu_sub,
            prior=prior,
            use_priors=use_priors
        )
        new_mus_sub_2, new_covs_sub_2, new_pis_2 = _create_subclusters(
            k_sub=2 * k + 1,
            codes=codes,
            logits=logits,
            logits_sub=logits_sub,
            mus_sub=mus_sub,
            pi_sub=pi_sub,
            n_sub=n_sub,
            how_to_init_mu_sub=how_to_init_mu_sub,
            prior=prior,
            use_priors=use_priors
        )
        mus_sub_to_add.extend([new_mus_sub_1, new_mus_sub_2])
        covs_sub_to_add.extend([new_covs_sub_1, new_covs_sub_2])
        pis_sub_to_add.extend([new_pis_1, new_pis_2])

    mus_sub_new = torch.cat([mus_sub_new, torch.cat(mus_sub_to_add)])
    covs_sub_new = torch.cat([covs_sub_new, torch.cat(covs_sub_to_add)])
    pi_sub_new = torch.cat([pi_sub_new, torch.cat(pis_sub_to_add)])

    return mus_sub_new, covs_sub_new, pi_sub_new

def update_models_parameters_split(
    split_decisions,
    mus,
    covs,
    pi,
    mus_sub,
    covs_sub,
    pi_sub,
    codes,
    logits,
    logits_sub,
    n_sub_list,
    n_sub,
    how_to_init_mu_sub,
    prior,
    use_priors,
    cluster_labels,
    subcluster_labels,
    prior_labels,
    accepted_subclusters=None  # <-- NEW!
):
    """
    Update main-cluster and subcluster parameters after a split.
    Now supports partial subcluster acceptance via accepted_subclusters.

    Returns:
    --------
    (
        mus_new, covs_new, pi_new,             # updated main cluster params
        mus_sub_new, covs_sub_new, pi_sub_new, # updated subcluster params
        prior_new, n_sub_list_new,            # updated prior + updated n_sub_list
        updated_cluster_labels,               # new dictionary for main clusters
        updated_subcluster_labels,            # new dictionary for subclusters
        updated_prior_labels,                 # new dictionary for prior labels
        origin_cardinality                    # <-- NEW: dictionary that maps
                                              #     new main cluster idx -> # of subclusters that formed it
    )
    """

    # ----------------------------
    # 1) Update main clusters
    # ----------------------------
    (
        mus_new,
        covs_new,
        pi_new,
        prior_new,
        updated_cluster_labels,
        updated_prior_labels,
        origin_cardinality  # <-- we retrieve this from update_clusters_params_split
    ) = update_clusters_params_split(
        mus,
        covs,
        pi,
        split_decisions,
        mus_sub,
        covs_sub,
        pi_sub,
        prior,
        codes,
        logits,
        logits_sub,
        n_sub_list,
        cluster_labels,
        prior_labels,
        accepted_subclusters=accepted_subclusters  # pass it down
    )

    # ----------------------------
    # 2) Update subclusters
    # ----------------------------
    (
        mus_sub_new,
        covs_sub_new,
        pi_sub_new,
        n_sub_list_new,
        updated_subcluster_labels
    ) = update_subclusters_params_split(
        mus_sub,
        covs_sub,
        pi_sub,
        split_decisions,
        codes,
        logits,
        logits_sub,
        n_sub_list,
        n_sub,
        how_to_init_mu_sub,
        prior_new,
        use_priors=use_priors,
        subcluster_labels=subcluster_labels,
        updated_cluster_labels=updated_cluster_labels,
        accepted_subclusters=accepted_subclusters  # pass it down
    )

    return (
        mus_new,
        covs_new,
        pi_new,
        mus_sub_new,
        covs_sub_new,
        pi_sub_new,
        prior_new,
        n_sub_list_new,
        updated_cluster_labels,
        updated_subcluster_labels,
        updated_prior_labels,
        origin_cardinality  # <-- pass it up
    )


def update_models_parameters_split_N_subcluster(
    split_decisions,
    mus,
    covs,
    pi,
    mus_sub,
    covs_sub,
    pi_sub,
    codes,
    logits,
    logits_sub,
    n_sub_list,
    n_sub,
    how_to_init_mu_sub,
    prior,
    use_priors,
    cluster_labels,
    subcluster_labels,
    prior_labels
):
    # Update clusters
    (
        mus_new,
        covs_new,
        pi_new,
        prior_new,
        updated_cluster_labels,
        updated_prior_labels
    ) = update_clusters_params_split(
        mus,
        covs,
        pi,
        split_decisions,
        mus_sub,
        covs_sub,
        pi_sub,
        prior,
        codes,
        logits,
        logits_sub,
        n_sub_list,
        cluster_labels,
        prior_labels
    )
    # Update subclusters
    (
        mus_sub_new,
        covs_sub_new,
        pi_sub_new,
        n_sub_list_new,
        updated_subcluster_labels
    ) = update_subclusters_params_split(
        mus_sub,
        covs_sub,
        pi_sub,
        split_decisions,
        codes,
        logits,
        logits_sub,
        n_sub_list,
        n_sub,
        how_to_init_mu_sub,
        prior_new,
        use_priors=use_priors,
        subcluster_labels=subcluster_labels,
        updated_cluster_labels=updated_cluster_labels  # Pass updated cluster labels
    )

    return (
        mus_new,
        covs_new,
        pi_new,
        mus_sub_new,
        covs_sub_new,
        pi_sub_new,
        prior_new,
        n_sub_list_new,
        updated_cluster_labels,
        updated_subcluster_labels,
        updated_prior_labels
    )



def update_models_parameters_split_2Sub(
    split_decisions,
    mus,
    covs,
    pi,
    mus_ind_to_split,
    mus_sub,
    covs_sub,
    pi_sub,
    codes,
    logits,
    logits_sub,
    n_sub,
    how_to_init_mu_sub,
    prior,
    use_priors
):
    mus_ind_to_split = torch.nonzero(split_decisions, as_tuple=False)
    # update the mus, covs and pis
    mus_new, covs_new, pi_new,prior_new = update_clusters_params_split(
        mus, covs, pi, mus_ind_to_split, split_decisions, mus_sub, covs_sub, pi_sub,prior,codes,logits,logits_sub
    )
    #print('SPLIT PHASE PRIOR NEW:',prior_new.mus_covs_prior.niw_psi_clusters.size())
    # update the submus, subcovs and subpis
    mus_sub_new, covs_sub_new, pi_sub_new = update_subclusters_params_split(
        mus_sub,
        covs_sub,
        pi_sub,
        mus_ind_to_split,
        split_decisions,
        codes,
        logits,
        logits_sub,
        n_sub,
        how_to_init_mu_sub,
        prior_new,
        use_priors=use_priors
    )
    return mus_new, covs_new, pi_new, mus_sub_new, covs_sub_new, pi_sub_new,prior_new




def update_clusters_params_merge(
    mus_lists_to_merge,
    inds_to_mask,
    mus,
    covs,
    pi,
    K,
    codes,
    logits,
    prior,
    use_priors,
    n_sub,
    how_to_init_mu_sub,
    clusters_to_suppress  # Add this parameter
):
    # Combine inds_to_mask and clusters_to_suppress
    total_inds_to_mask = inds_to_mask.clone()
    for idx in clusters_to_suppress:
        total_inds_to_mask[idx] = True

    # Get clusters not merged or suppressed
    mus_not_merged = mus[~total_inds_to_mask]
    covs_not_merged = covs[~total_inds_to_mask]
    pis_not_merged = pi[~total_inds_to_mask]

    # Lists to store updated parameters and psi priors
    mus_merged, covs_merged, pi_merged, psi_new_list = [], [], [], []

    for group in mus_lists_to_merge:
        # Get counts N_k for each cluster in the group
        N_k_list = [(logits.argmax(-1) == k).sum().type(torch.float32) for k in group]
        N_k_total = sum(N_k_list)
        codes_k_list=[]
        mus_list = [mus[k] for k in group]
        predicted_classes = logits.argmax(dim=-1)  # shape: (…,) of ints
        codes_k_list = [codes[predicted_classes == k] for k in group]


        if N_k_total > 0:
            # Compute the merged mean using the function with N_k_list
            #merged_mean = compute_merged_mean(mus_list, N_k_list)
            merged_mean = compute_merged_mean_karcher(codes_k_list)
        else:
            # Handle empty clusters: use equal weights
            #merged_mean = compute_merged_mean(mus_list, None)
            fallback_codes = torch.stack([mus[k] for k in cluster_indices], dim=0)  # (C, D)
            merged_mean = KarcherMean(soft_assign=None, codes=fallback_codes, cov=None)

        # Compute combined responsibilities for the merged cluster
        logits_k = logits[:, group].sum(dim=1).reshape(-1, 1)
        cov_new = compute_data_covs_soft_assignment(
            logits=logits_k,
            codes=codes,
            K=1,
            mus=merged_mean.unsqueeze(0),
            prior_name=prior.name
        )

        # Compute the psi prior for the merged cluster
        if prior.prior_choice == 'dynamic_data_std':
            indices_k = torch.isin(logits.argmax(-1), torch.tensor(group, device=logits.device))
            codes_k = codes[indices_k]
            if codes_k.size(0) > 0:
                # Recompute psi_k based on the data points in the merged cluster
                log_map_k = Log_mapping(codes_k, merged_mean)
                std_k = log_map_k.std(axis=0)
                psi_k = torch.diag(std_k) * prior.get_prior_sigma_scale()
                psi_k = psi_k.double()
                print(f"Computed psi_k for merged cluster based on merged data.")
            else:
                # Use psi_k from one of the original clusters
                psi_k = prior.mus_covs_prior.niw_psi_clusters[group[0]]
                print(f"No data points in merged cluster; using psi_k from cluster {group[0]}.")

            if psi_k.dim() == 2:
                psi_k = psi_k.unsqueeze(0)
            psi_new_list.append(psi_k)
        else:
            # Handle other priors if necessary
            pass

        # Update mixing coefficients
        pi_new = sum(pi[k] for k in group).reshape(1)

        D = codes.size(1)
        r_k = logits[:, group].sum()
        
        if use_priors:
            if prior.prior_choice == 'dynamic_data_std':
                cov_new = prior.compute_post_cov(r_k, cov_new, D, psi_index=None, psi_value=psi_k)
            else:
                cov_new = prior.compute_post_cov(r_k, cov_new, D)
        else:
            if not positive_definite.check(cov_new):
                print("[WARNING] cov_new is not positive definite. Adjusting...")
                cov_new = ensure_positive_definite(cov_new)
            cov_new = cov_new.to(codes.device)

        

        mus_merged.append(merged_mean)
        covs_merged.append(cov_new)
        pi_merged.append(pi_new)

    # Stack the merged parameters
    if mus_merged:
        mus_merged = torch.stack(mus_merged).squeeze(1)
        covs_merged = torch.stack(covs_merged).squeeze(1)
        pi_merged = torch.stack(pi_merged).squeeze(1)
    else:
        mus_merged = torch.empty((0, mus.size(1)), device=mus.device)
        covs_merged = torch.empty((0, covs.size(1), covs.size(2)), device=covs.device)
        pi_merged = torch.empty((0,), device=pi.device)

    # Concatenate non-merged and merged parameters
    mus_new = torch.cat([mus_not_merged, mus_merged], dim=0)
    covs_new = torch.cat([covs_not_merged, covs_merged], dim=0)
    pi_new = torch.cat([pis_not_merged, pi_merged], dim=0)

    # Update the prior object
    if prior.prior_choice == 'dynamic_data_std':
        # Exclude suppressed clusters from prior
        prior_new_psi_clusters = prior.mus_covs_prior.niw_psi_clusters[~total_inds_to_mask]
        if psi_new_list:
            prior_new_psi_clusters = torch.cat([prior_new_psi_clusters] + psi_new_list, dim=0)
        prior.mus_covs_prior.niw_psi_clusters = prior_new_psi_clusters

    prior_new = prior

    return mus_new, covs_new, pi_new, prior_new



def update_clusters_params_merge_2sub(
    mus_lists_to_merge,
    inds_to_mask,
    mus,
    covs,
    pi,
    K,
    codes,
    logits,
    prior,
    use_priors,
    n_sub,
    how_to_init_mu_sub,
):
    mus_not_merged = mus[torch.logical_not(inds_to_mask)]
    covs_not_merged = covs[torch.logical_not(inds_to_mask)]
    pis_not_merged = pi[torch.logical_not(inds_to_mask)]

    # Lists to store updated parameters and psi priors
    mus_merged, covs_merged, pi_merged, psi_new_list = [], [], [], []

    for pair in mus_lists_to_merge:
        N_k_1 = (logits.argmax(-1) == pair[0]).sum().type(torch.float32)
        N_k_2 = (logits.argmax(-1) == pair[1]).sum().type(torch.float32)
        N_k = N_k_1 + N_k_2

        if N_k > 0:
            percentage = N_k_2 / N_k
            mus_mean = rotate_vector_a_to_b(mus[pair[0]], mus[pair[1]], percentage=percentage)
            D = codes.size(1)

            # Compute combined responsibilities for the merged cluster
            logits_k = (logits[:, pair[0]] + logits[:, pair[1]]).reshape(-1, 1)
            cov_new = compute_data_covs_soft_assignment(
                logits=logits_k,
                codes=codes,
                K=1,
                mus=mus_mean,
                prior_name=prior.name
            )

            # Compute the psi prior for the merged cluster
            if prior.prior_choice == 'dynamic_data_std':
                # Get indices of data points assigned to the merged cluster
                indices_k = (logits.argmax(-1) == pair[0]) | (logits.argmax(-1) == pair[1])
                codes_k = codes[indices_k]

                if codes_k.size(0) > 0:
                    # Compute Karcher mean of codes_k
                    karcher_mean_k = KarcherMean(None, codes_k)
                    # Compute log mapping
                    log_map_k = Log_mapping(codes_k, karcher_mean_k)
                    # Compute standard deviation
                    std_k = log_map_k.std(axis=0)
                    # Compute psi prior
                    psi_k = torch.diag(std_k) * prior.get_prior_sigma_scale()
                    psi_k = psi_k.double()
                else:
                    # If no data points, average psi of original clusters
                    psi_k = (prior.mus_covs_prior.niw_psi_clusters[pair[0]] + prior.mus_covs_prior.niw_psi_clusters[pair[1]]) / 2
                if psi_k.dim()==2:
                   psi_k=psi_k.unsqueeze(0)
                psi_new_list.append(psi_k)
        else:
            # In case both clusters are empty
            mus_mean = rotate_vector_a_to_b(mus[pair[0]], mus[pair[1]], percentage=0.5)
            cov_new = covs[pair[0]].unsqueeze(0)
            # Average the psi priors of the original clusters
            psi_k = (prior.mus_covs_prior.niw_psi_clusters[pair[0]] + prior.mus_covs_prior.niw_psi_clusters[pair[1]]) / 2
            psi_new_list.append(psi_k)

        pi_new = (pi[pair[0]] + pi[pair[1]]).reshape(1)

        D = codes.size(1)
        r_k = (logits[:, pair[0]] + logits[:, pair[1]]).sum()
        
        if use_priors:
            if prior.prior_choice == 'dynamic_data_std':
                cov_new = prior.compute_post_cov(r_k, cov_new, D, psi_index=None, psi_value=psi_k)
            else:
                cov_new = prior.compute_post_cov(r_k, cov_new, D)
        else:
            if not positive_definite.check(cov_new):
                print("[WARNING] cov_new (merged pair) is not positive definite. Adjusting...")
                cov_new = ensure_positive_definite(cov_new)
            cov_new = cov_new.to(codes.device)


        print('cov in update_clusters_params_merge: ', cov_new)
        mus_merged.append(mus_mean)
        covs_merged.append(cov_new)
        pi_merged.append(pi_new)

    # Stack the merged parameters
    mus_merged = torch.stack(mus_merged).squeeze(1)
    covs_merged = torch.stack(covs_merged).squeeze(1)
    pi_merged = torch.stack(pi_merged).squeeze(1)

    # Concatenate non-merged and merged parameters
    mus_new = torch.cat([mus_not_merged, mus_merged])
    covs_new = torch.cat([covs_not_merged, covs_merged])
    pi_new = torch.cat([pis_not_merged, pi_merged])

   

    # Update the prior object
    if prior.prior_choice == 'dynamic_data_std':
      # Update the prior's psi clusters
      prior_new_psi_clusters = prior.mus_covs_prior.niw_psi_clusters[torch.logical_not(inds_to_mask)]
      prior_new_psi_clusters = torch.cat([prior_new_psi_clusters] + psi_new_list)
      prior.mus_covs_prior.niw_psi_clusters=prior_new_psi_clusters
    
    prior_new = prior

    return mus_new, covs_new, pi_new, prior_new


def update_subclusters_params_merge_DPM(
    mus_lists_to_merge, inds_to_mask, mus, covs, pi, mus_sub, covs_sub, pi_sub,
    codes, logits, n_sub, how_to_init_mu_sub, prior, use_priors=True
):
    # update sub_mus
    mus_sub_not_merged = mus_sub[torch.logical_not(inds_to_mask.repeat_interleave(2))]
    covs_sub_not_merged = covs_sub[torch.logical_not(inds_to_mask.repeat_interleave(2))]
    pi_sub_not_merged = pi_sub[torch.logical_not(inds_to_mask.repeat_interleave(2))]

    mus_sub_merged, covs_sub_merged, pi_sub_merged = [], [], []
    for n_merged in range(len(mus_lists_to_merge)):
        codes_merged = codes[torch.logical_or((logits.argmax(-1) == mus_lists_to_merge[n_merged][0]), (logits.argmax(-1) == mus_lists_to_merge[n_merged][1]))]
        if len(codes_merged) <= 5:
            # Both clusters are empty or have very few points
            mus_sub_merged.append(mus[mus_lists_to_merge[n_merged].flatten()])
            covs_sub_merged.append(covs[mus_lists_to_merge[n_merged].flatten()])
            pi_sub_merged.append(pi[mus_lists_to_merge[n_merged].flatten()])
        else:
            mus_sub_k, covs_sub_k, pi_sub_k = init_mus_and_covs_sub(codes_merged, k=0, n_sub=n_sub, how_to_init_mu_sub=how_to_init_mu_sub, logits=torch.zeros(len(codes_merged), 1), logits_sub=None, prior=prior, use_priors=use_priors, device=codes.device)
            mus_sub_merged.append(mus_sub_k)
            covs_sub_merged.append(covs_sub_k)
            pi_sub_merged.append(pi_sub_k)
    mus_sub_merged = torch.cat(mus_sub_merged)
    covs_sub_merged = torch.cat(covs_sub_merged)
    pi_sub_merged = torch.cat(pi_sub_merged)

    mus_sub_new = torch.cat([mus_sub_not_merged, mus_sub_merged])
    covs_sub_new = torch.cat([covs_sub_not_merged, covs_sub_merged])
    pi_sub_new = torch.cat([pi_sub_not_merged, pi_sub_merged])

    return mus_sub_new, covs_sub_new, pi_sub_new


def update_subclusters_params_merge(
    mus_lists_to_merge, inds_to_mask, mus, covs, pi, mus_sub, covs_sub, pi_sub,
    codes, logits, n_sub, how_to_init_mu_sub, prior, mus_new, use_priors=True,
    n_sub_list=None, clusters_to_suppress=None  # Add this parameter
):
    device = codes.device  # Get the device from the `codes` tensor

    # Combine inds_to_mask and clusters_to_suppress
    total_inds_to_mask = inds_to_mask.clone()
    for idx in clusters_to_suppress:
        total_inds_to_mask[idx] = True

    # Compute n_sub_list for non-merged and non-suppressed clusters
    n_sub_list_not_merged = [n_sub_list[i] for i in range(len(total_inds_to_mask)) if not total_inds_to_mask[i]]

    # Indices of subclusters not merged or suppressed
    subcluster_indices_not_merged = []
    current_sub_idx = 0
    for i in range(len(total_inds_to_mask)):
        num_subclusters = n_sub_list[i]
        if not total_inds_to_mask[i]:
            subcluster_indices_not_merged.extend(range(current_sub_idx, current_sub_idx + num_subclusters))
        current_sub_idx += num_subclusters

    subcluster_indices_not_merged = torch.tensor(subcluster_indices_not_merged, dtype=torch.long)

    mus_sub_not_merged = mus_sub[subcluster_indices_not_merged].to(device=device)
    covs_sub_not_merged = covs_sub[subcluster_indices_not_merged].to(device=device)
    pi_sub_not_merged = pi_sub[subcluster_indices_not_merged].to(device=device)

    mus_not_merged = mus[~total_inds_to_mask]
    num_not_merged = mus_not_merged.size(0)

    mus_sub_merged, covs_sub_merged, pi_sub_merged = [], [], []
    n_sub_list_new = n_sub_list_not_merged.copy()

    for n_merged, group in enumerate(mus_lists_to_merge):
        # Get data points assigned to the clusters being merged
        indices_k = torch.isin(logits.argmax(-1), torch.tensor(group, device=logits.device))
        codes_merged = codes[indices_k]
        merged_cluster_index = num_not_merged + n_merged
        mus_current = mus_new[merged_cluster_index].unsqueeze(0)

        if len(codes_merged) <= 5:
            # Handle case when clusters have very few points
            # Use the subclusters of the cluster with highest log-likelihood
            highest_ll_cluster = highest_ll_mus[n_merged]
            # Get the number of subclusters for the highest_ll_cluster
            num_subclusters_k = n_sub_list[highest_ll_cluster]
            # Get indices in mus_sub, covs_sub, pi_sub
            start_idx = sum(n_sub_list[:highest_ll_cluster])
            end_idx = start_idx + num_subclusters_k
            mus_sub_k = mus_sub[start_idx:end_idx].to(device)
            covs_sub_k = covs_sub[start_idx:end_idx].to(device)
            pi_sub_k = pi_sub[start_idx:end_idx].to(device)

            mus_sub_merged.append(mus_sub_k)
            covs_sub_merged.append(covs_sub_k)
            pi_sub_merged.append(pi_sub_k)
            # Update n_sub_list_new
            n_sub_list_new.append(num_subclusters_k)
        else:
            # Initialize subclusters for the merged cluster
            K = mus_new.size(0)
            logits_merged = torch.zeros((len(codes_merged), K)).to(device)
            logits_merged[:, merged_cluster_index] = 1  # Set the highest value for the selected cluster
            mus_sub_k, covs_sub_k, pi_sub_k,nb_sub = init_mus_and_covs_sub(
                codes_merged, k=merged_cluster_index, mus=mus_current, n_sub=n_sub,
                how_to_init_mu_sub=how_to_init_mu_sub, logits=logits_merged.to(device),
                logits_sub=None, prior=prior, use_priors=use_priors, device=device
            )

            # Ensure tensors are correctly formed
            if isinstance(mus_sub_k, list):
                mus_sub_k = torch.stack(mus_sub_k, dim=0).to(device)
            if isinstance(covs_sub_k, list):
                covs_sub_k = torch.stack(covs_sub_k, dim=0).to(device)
            if isinstance(pi_sub_k, list):
                pi_sub_k = torch.stack(pi_sub_k, dim=0).to(device)

            mus_sub_merged.append(mus_sub_k)
            covs_sub_merged.append(covs_sub_k)
            pi_sub_merged.append(pi_sub_k)
            # Update n_sub_list_new
            n_sub_list_new.append(nb_sub)

    # Concatenate the merged tensors
    if mus_sub_merged:
        mus_sub_merged = torch.cat(mus_sub_merged, dim=0).to(device)
        covs_sub_merged = torch.cat(covs_sub_merged, dim=0).to(device)
        pi_sub_merged = torch.cat(pi_sub_merged, dim=0).to(device)
    else:
        mus_sub_merged = torch.empty((0, mus_sub.size(1)), device=device)
        covs_sub_merged = torch.empty((0, covs_sub.size(1), covs_sub.size(2)), device=device)
        pi_sub_merged = torch.empty((0,), device=device)

    # Concatenate the non-merged and merged components
    mus_sub_new = torch.cat([mus_sub_not_merged, mus_sub_merged], dim=0)
    covs_sub_new = torch.cat([covs_sub_not_merged, covs_sub_merged], dim=0)
    pi_sub_new = torch.cat([pi_sub_not_merged, pi_sub_merged], dim=0)

    return mus_sub_new, covs_sub_new, pi_sub_new, n_sub_list_new



def update_subclusters_params_merge_2sub(
    mus_lists_to_merge, inds_to_mask, mus, covs, pi, mus_sub, covs_sub, pi_sub,
    codes, logits, n_sub, how_to_init_mu_sub, prior,mus_new, use_priors=True
):
    device = codes.device  # Get the device from the `codes` tensor

    # Move non-merged tensors to the same device as `codes`
    mus_sub_not_merged = mus_sub[torch.logical_not(inds_to_mask.repeat_interleave(2))].to(device=device)
    covs_sub_not_merged = covs_sub[torch.logical_not(inds_to_mask.repeat_interleave(2))].to(device=device)
    pi_sub_not_merged = pi_sub[torch.logical_not(inds_to_mask.repeat_interleave(2))].to(device=device)
    
    mus_not_merged = mus[torch.logical_not(inds_to_mask)]
    num_not_merged = mus_not_merged.size(0)  # Number of non-merged clusters

    mus_sub_merged, covs_sub_merged, pi_sub_merged = [], [], []
    
    for n_merged in range(len(mus_lists_to_merge)):
        codes_merged = codes[torch.logical_or(
            (logits.argmax(-1) == mus_lists_to_merge[n_merged][0]),
            (logits.argmax(-1) == mus_lists_to_merge[n_merged][1])
        )]
        merged_cluster_index = num_not_merged + n_merged
        mus_current = mus_new[merged_cluster_index].unsqueeze(0)
        if len(codes_merged) <= 5:
            # Handle case when clusters have very few points
            mus_sub_merged.append(mus[mus_lists_to_merge[n_merged].flatten()].to(device=device))
            covs_sub_merged.append(covs[mus_lists_to_merge[n_merged].flatten()].to(device=device))
            pi_sub_merged.append(pi[mus_lists_to_merge[n_merged].flatten()].to(device=device))
        else:
            K = mus_new.size(0)
            logits_merged = torch.zeros((len(codes_merged), K)).to(device)
            logits_merged[:, merged_cluster_index] = 1  # Set the highest value for the selected cluster
            mus_sub_k, covs_sub_k, pi_sub_k = init_mus_and_covs_sub(
                codes_merged, k=merged_cluster_index, mus=mus_current, n_sub=n_sub, 
                how_to_init_mu_sub=how_to_init_mu_sub, logits=logits_merged.to(device), 
                logits_sub=None, prior=prior, use_priors=use_priors, device=device
            )

            # Handle case where mus_sub_k, covs_sub_k, and pi_sub_k are lists of tensors
            if isinstance(mus_sub_k, list):
                mus_sub_k = torch.stack(mus_sub_k, dim=0).to(device)  # Stack into tensor
            if isinstance(covs_sub_k, list):
                covs_sub_k = torch.stack(covs_sub_k, dim=0).to(device)  # Stack into tensor
            if isinstance(pi_sub_k, list):
                pi_sub_k = torch.stack(pi_sub_k, dim=0).to(device)  # Stack into tensor

            mus_sub_merged.append(mus_sub_k)
            covs_sub_merged.append(covs_sub_k)
            pi_sub_merged.append(pi_sub_k)

    # Concatenate the merged tensors
    mus_sub_merged = torch.cat(mus_sub_merged).to(device)
    covs_sub_merged = torch.cat(covs_sub_merged).to(device)
    pi_sub_merged = torch.cat(pi_sub_merged).to(device)

    # Concatenate the non-merged and merged components (both on the same device)
    mus_sub_new = torch.cat([mus_sub_not_merged, mus_sub_merged], dim=0)
    covs_sub_new = torch.cat([covs_sub_not_merged, covs_sub_merged], dim=0)
    pi_sub_new = torch.cat([pi_sub_not_merged, pi_sub_merged], dim=0)

    return mus_sub_new, covs_sub_new, pi_sub_new
    

def update_subclusters_params_merge_v1(
    mus_lists_to_merge, inds_to_mask, mus, covs, pi, mus_sub, covs_sub, pi_sub,
    codes, logits, n_sub, how_to_init_mu_sub, prior,mus_new, use_priors=True
):
    device = codes.device  # Get the device from the `codes` tensor

    # Move non-merged tensors to the same device as `codes`
    mus_sub_not_merged = mus_sub[torch.logical_not(inds_to_mask.repeat_interleave(2))].to(device=device)
    covs_sub_not_merged = covs_sub[torch.logical_not(inds_to_mask.repeat_interleave(2))].to(device=device)
    pi_sub_not_merged = pi_sub[torch.logical_not(inds_to_mask.repeat_interleave(2))].to(device=device)

    mus_sub_merged, covs_sub_merged, pi_sub_merged = [], [], []
    
    for n_merged in range(len(mus_lists_to_merge)):
        codes_merged = codes[torch.logical_or(
            (logits.argmax(-1) == mus_lists_to_merge[n_merged][0]),
            (logits.argmax(-1) == mus_lists_to_merge[n_merged][1])
        )]

        if len(codes_merged) <= 5:
            # Handle case when clusters have very few points
            mus_sub_merged.append(mus[mus_lists_to_merge[n_merged].flatten()].to(device=device))
            covs_sub_merged.append(covs[mus_lists_to_merge[n_merged].flatten()].to(device=device))
            pi_sub_merged.append(pi[mus_lists_to_merge[n_merged].flatten()].to(device=device))
        else:
            mus_sub_k, covs_sub_k, pi_sub_k = init_mus_and_covs_sub(
                codes_merged, k=0, mus=mus_new[n_merged].unsqueeze(0), n_sub=n_sub, 
                how_to_init_mu_sub=how_to_init_mu_sub, logits=torch.zeros(len(codes_merged), 1).to(device), 
                logits_sub=None, prior=prior, use_priors=use_priors, device=device
            )

            # Handle case where mus_sub_k, covs_sub_k, and pi_sub_k are lists of tensors
            if isinstance(mus_sub_k, list):
                mus_sub_k = torch.stack(mus_sub_k, dim=0).to(device)  # Stack into tensor
            if isinstance(covs_sub_k, list):
                covs_sub_k = torch.stack(covs_sub_k, dim=0).to(device)  # Stack into tensor
            if isinstance(pi_sub_k, list):
                pi_sub_k = torch.stack(pi_sub_k, dim=0).to(device)  # Stack into tensor

            mus_sub_merged.append(mus_sub_k)
            covs_sub_merged.append(covs_sub_k)
            pi_sub_merged.append(pi_sub_k)

    # Concatenate the merged tensors
    mus_sub_merged = torch.cat(mus_sub_merged).to(device)
    covs_sub_merged = torch.cat(covs_sub_merged).to(device)
    pi_sub_merged = torch.cat(pi_sub_merged).to(device)

    # Concatenate the non-merged and merged components (both on the same device)
    mus_sub_new = torch.cat([mus_sub_not_merged, mus_sub_merged], dim=0)
    covs_sub_new = torch.cat([covs_sub_not_merged, covs_sub_merged], dim=0)
    pi_sub_new = torch.cat([pi_sub_not_merged, pi_sub_merged], dim=0)

    return mus_sub_new, covs_sub_new, pi_sub_new





def update_models_parameters_merge(
    mus_lists_to_merge,
    inds_to_mask,
    K,
    mus,
    covs,
    pi,
    mus_sub,
    covs_sub,
    pi_sub,
    codes,
    logits,
    prior,
    use_priors,
    n_sub,
    how_to_init_mu_sub,
    n_sub_list,
    clusters_to_suppress,  # Add this parameter
):
    # Update clusters
    mus_new, covs_new, pi_new, prior = update_clusters_params_merge(
        mus_lists_to_merge,
        inds_to_mask,
        mus,
        covs,
        pi,
        K,
        codes,
        logits,
        prior,
        use_priors,
        n_sub,
        how_to_init_mu_sub,
        clusters_to_suppress  # Pass suppressed clusters
    )
    # Update subclusters
    mus_sub_new, covs_sub_new, pi_sub_new, n_sub_list_new = update_subclusters_params_merge(
        mus_lists_to_merge, inds_to_mask, mus, covs, pi, mus_sub, covs_sub, pi_sub, codes, logits,
        n_sub, how_to_init_mu_sub, prior, mus_new, use_priors=use_priors, n_sub_list=n_sub_list,
        clusters_to_suppress=clusters_to_suppress  # Pass suppressed clusters
    )
    return mus_new, covs_new, pi_new, mus_sub_new, covs_sub_new, pi_sub_new, prior, n_sub_list_new



def update_models_parameters_merge_2sub(
    mus_lists_to_merge,
    inds_to_mask,
    K,
    mus,
    covs,
    pi,
    mus_sub,
    covs_sub,
    pi_sub,
    codes,
    logits,
    prior,
    use_priors,
    n_sub, how_to_init_mu_sub,
):

    mus_new, covs_new, pi_new , prior = update_clusters_params_merge(
        mus_lists_to_merge,
        inds_to_mask,
        mus,
        covs,
        pi,
        K,
        codes,
        logits,
        prior,
        use_priors,
        n_sub,
        how_to_init_mu_sub,
    )
    print('MUS NEW' , mus_new)
    print('mus new size' , mus_new.size())
    print('mus_lists_to_merge',mus_lists_to_merge)
    mus_sub_new, covs_sub_new, pi_sub_new = update_subclusters_params_merge(
        mus_lists_to_merge, inds_to_mask, mus, covs, pi, mus_sub, covs_sub, pi_sub, codes, logits, n_sub, how_to_init_mu_sub, prior, mus_new, use_priors=use_priors
    )
    return mus_new, covs_new, pi_new, mus_sub_new, covs_sub_new, pi_sub_new , prior



def merge_rule(mus, logits, codes, cluster_indices, alpha, cov_const, merge_prob, prior=None, covs=None):
    """
    Determines whether to merge a group of clusters based on the generalized Hastings ratio.

    Args:
        mus (torch.Tensor): Cluster centers.
        logits (torch.Tensor): Responsibilities or soft assignments.
        codes (torch.Tensor): Data points.
        cluster_indices (list): List of cluster indices to consider merging.
        alpha (float): Concentration parameter.
        cov_const, merge_prob: Hyperparameters.
        prior: Prior object.
        covs (torch.Tensor): Covariance matrices of clusters.

    Returns:
        decisions (list): List containing a single boolean indicating the merge decision.
        highest_ll (int): Index of the cluster with the highest log-likelihood among the group.
    """
    # Initialize lists to collect statistics
    N_k_list = []
    codes_k_list = []
    log_ll_k_list = []
    q_RandomParamProposalt_list = []
    all_empty = True

    # Total data points in the merged cluster
    N_c = 0

    print(f"\n[merge_rule] Attempting to merge clusters: {cluster_indices}")

    # Collect data points and compute log-likelihoods for each cluster
    for k in cluster_indices:
        codes_ind_k = logits.argmax(-1) == k
        codes_k = codes[codes_ind_k]
        N_k = len(codes_k)
        N_k_list.append(N_k)
        N_c += N_k
        codes_k_list.append(codes_k)
        if N_k > 0:
            all_empty = False
        print(f"Cluster {k}: N_k = {N_k}")
        
        if prior.prior_choice == 'dynamic_data_std':
            log_ll_k = prior.log_marginal_likelihood_dynamic_psi(codes_k, covs[k], k)
        else:
            log_ll_k = prior.log_marginal_likelihood(codes_k, covs[k])

        log_ll_k_list.append(log_ll_k)

        print(f"Cluster {k}: log_ll_k = {log_ll_k}")

        q_RandomParamProposalt = prior.log_mean_projected_data(codes_k,covs[k])
        q_RandomParamProposalt_list.append(q_RandomParamProposalt)

        print(f"Cluster {k}: q_RandomParamProposalt = {q_RandomParamProposalt}")
    
    if all_empty:
        print(f"All clusters {cluster_indices} are empty. Merging directly.")
        decision = True
        highest_ll_index = cluster_indices[0]  # Arbitrarily choose one
        return [decision], highest_ll_index

    # Combine all data points for the merged cluster
    codes_k_merged = torch.cat(codes_k_list, dim=0)

    # Compute the merged mean
    #mus_list = [mus[k] for k in cluster_indices]
    #mus_mean = compute_merged_mean(mus_list, N_k_list)
    mus_mean = compute_merged_mean_karcher(codes_k_list)
    

    print(f"Merged mean computed for clusters {cluster_indices}.")

    # Compute the covariance for the merged cluster
    logits_k = logits[:, cluster_indices].sum(dim=1).reshape(-1, 1)
    covs_mean = compute_data_covs_soft_assignment(
        logits=logits_k,
        codes=codes,
        K=1,
        mus=mus_mean.unsqueeze(0),
        prior_name=prior.name
    )

    print(f"Merged covariance computed.")

    # Compute the prior for the merged cluster
    if prior.prior_choice == 'dynamic_data_std':
        if codes_k_merged.size(0) > 0:
            # Recompute psi_k based on the data points in the merged cluster
            log_map_k = Log_mapping(codes_k_merged, mus_mean)
            std_k = log_map_k.std(axis=0)
            psi_k = torch.diag(std_k) * prior.get_prior_sigma_scale()
            psi_k = psi_k.double()
            print(f"Computed psi_k for merged cluster based on merged data.")
        else:
            # Handle the case with no data points (e.g., use default psi)
            psi_k = prior.mus_covs_prior.niw_psi_clusters[cluster_indices[0]]
            print(f"No data points in merged cluster; using psi_k from cluster {cluster_indices[0]}.")
        
        if psi_k.dim() == 2:
            psi_k = psi_k.unsqueeze(0)
        
        # Compute the posterior covariance using the new psi_k
        covs_mean = prior.compute_post_cov(logits_k.sum(), covs_mean, codes.size(1), psi_index=None, psi_value=psi_k)
        
        # Compute the log-likelihood for the merged cluster
        log_ll_k_merged = prior.log_marginal_likelihood_dynamic_psi(codes_k_merged, covs_mean, psi_index=None, psi_value=psi_k)
    else:
        # For other priors, use existing methods
        covs_mean = prior.compute_post_cov(logits_k.sum(), covs_mean, codes.size(1))
        log_ll_k_merged = prior.log_marginal_likelihood(codes_k_merged, covs_mean)

    print(f"Merged cluster log likelihood: {log_ll_k_merged}")

    # Compute q for the merged cluster
    q_RandomParamProposalt_merged = prior.log_mean_projected_data(codes_k_merged,covs_mean)

    print(f"Merged cluster q_RandomParamProposalt: {q_RandomParamProposalt_merged}")

    # Determine the highest log-likelihood among individual clusters
    log_ll_k_tensor = torch.tensor(log_ll_k_list)
    highest_ll_index = cluster_indices[torch.argmax(log_ll_k_tensor)]

    print(f"Highest log-likelihood among clusters is at cluster {highest_ll_index}")

    # Compute the generalized Hastings ratio
    decision = log_Hastings_ratio_merge(
        alpha, N_k_list, log_ll_k_list, N_c, log_ll_k_merged, merge_prob,
        q_RandomParamProposalt_list, q_RandomParamProposalt_merged
    )

    print(f"Merge decision for clusters {cluster_indices}: {decision}")

    return [decision], highest_ll_index
    
def merge_rule_previous(mus, logits, codes, cluster_indices, alpha, cov_const, merge_prob, prior=None, covs=None):
    """
    Determines whether to merge a group of clusters based on the generalized Hastings ratio.

    Args:
        mus (torch.Tensor): Cluster centers.
        logits (torch.Tensor): Responsibilities or soft assignments.
        codes (torch.Tensor): Data points.
        cluster_indices (list): List of cluster indices to consider merging.
        alpha (float): Concentration parameter.
        cov_const, merge_prob: Hyperparameters.
        prior: Prior object.
        covs (torch.Tensor): Covariance matrices of clusters.

    Returns:
        decisions (list): List containing a single boolean indicating the merge decision.
        highest_ll (int): Index of the cluster with the highest log-likelihood among the group.
    """
    # Initialize lists to collect statistics
    N_k_list = []
    codes_k_list = []
    log_ll_k_list = []
    q_RandomParamProposalt_list = []
    all_empty = True

    # Total data points in the merged cluster
    N_c = 0

    print(f"\n[merge_rule] Attempting to merge clusters: {cluster_indices}")

    # Collect data points and compute log-likelihoods for each cluster
    for k in cluster_indices:
        codes_ind_k = logits.argmax(-1) == k
        codes_k = codes[codes_ind_k]
        N_k = len(codes_k)
        N_k_list.append(N_k)
        N_c += N_k
        codes_k_list.append(codes_k)
        if N_k > 0:
            all_empty = False
        print(f"Cluster {k}: N_k = {N_k}")
        
        if prior.prior_choice == 'dynamic_data_std':
            log_ll_k = prior.log_marginal_likelihood_dynamic_psi(codes_k, covs[k], k)
        else:
            log_ll_k = prior.log_marginal_likelihood(codes_k, covs[k])

        log_ll_k_list.append(log_ll_k)

        print(f"Cluster {k}: log_ll_k = {log_ll_k}")

        q_RandomParamProposalt = prior.log_mean_projected_data(codes_k, mus[k], covs[k])
        q_RandomParamProposalt_list.append(q_RandomParamProposalt)

        print(f"Cluster {k}: q_RandomParamProposalt = {q_RandomParamProposalt}")
    
    if all_empty:
        print(f"All clusters {cluster_indices} are empty. Merging directly.")
        decision = True
        highest_ll_index = cluster_indices[0]  # Arbitrarily choose one
        return [decision], highest_ll_index

    # Combine all data points for the merged cluster
    codes_k_merged = torch.cat(codes_k_list, dim=0)

    # Compute the merged mean
    #mus_list = [mus[k] for k in cluster_indices]
    #mus_mean = compute_merged_mean(mus_list, N_k_list)
    mus_mean = compute_merged_mean_karcher(codes_k_list)
    print(f"Merged mean computed for clusters {cluster_indices}.")

    # Compute the covariance for the merged cluster
    logits_k = logits[:, cluster_indices].sum(dim=1).reshape(-1, 1)
    covs_mean = compute_data_covs_soft_assignment(
        logits=logits_k,
        codes=codes,
        K=1,
        mus=mus_mean.unsqueeze(0),
        prior_name=prior.name
    )

    print(f"Merged covariance computed.")

    # Compute the prior for the merged cluster
    if prior.prior_choice == 'dynamic_data_std':
        if codes_k_merged.size(0) > 0:
            # Recompute psi_k based on the data points in the merged cluster
            log_map_k = Log_mapping(codes_k_merged, mus_mean)
            std_k = log_map_k.std(axis=0)
            psi_k = torch.diag(std_k) * prior.get_prior_sigma_scale()
            psi_k = psi_k.double()
            print(f"Computed psi_k for merged cluster based on merged data.")
        else:
            # Handle the case with no data points (e.g., use default psi)
            psi_k = prior.mus_covs_prior.niw_psi_clusters[cluster_indices[0]]
            print(f"No data points in merged cluster; using psi_k from cluster {cluster_indices[0]}.")
        
        if psi_k.dim() == 2:
            psi_k = psi_k.unsqueeze(0)
        
        # Compute the posterior covariance using the new psi_k
        covs_mean = prior.compute_post_cov(logits_k.sum(), covs_mean, codes.size(1), psi_index=None, psi_value=psi_k)
        
        # Compute the log-likelihood for the merged cluster
        log_ll_k_merged = prior.log_marginal_likelihood_dynamic_psi(codes_k_merged, covs_mean, psi_index=None, psi_value=psi_k)
    else:
        # For other priors, use existing methods
        covs_mean = prior.compute_post_cov(logits_k.sum(), covs_mean, codes.size(1))
        log_ll_k_merged = prior.log_marginal_likelihood(codes_k_merged, covs_mean)

    print(f"Merged cluster log likelihood: {log_ll_k_merged}")

    # Compute q for the merged cluster
    q_RandomParamProposalt_merged = prior.log_mean_projected_data(codes_k_merged, mus_mean, covs_mean)

    print(f"Merged cluster q_RandomParamProposalt: {q_RandomParamProposalt_merged}")

    # Determine the highest log-likelihood among individual clusters
    log_ll_k_tensor = torch.tensor(log_ll_k_list)
    highest_ll_index = cluster_indices[torch.argmax(log_ll_k_tensor)]

    print(f"Highest log-likelihood among clusters is at cluster {highest_ll_index}")

    # Compute the generalized Hastings ratio
    decision = log_Hastings_ratio_merge(
        alpha, N_k_list, log_ll_k_list, N_c, log_ll_k_merged, merge_prob,
        q_RandomParamProposalt_list, q_RandomParamProposalt_merged
    )

    print(f"Merge decision for clusters {cluster_indices}: {decision}")

    return [decision], highest_ll_index



def merge_step(
    mus,
    logits,
    codes,
    K,
    raise_merge_proposals,
    cov_const,
    alpha,
    merge_prob,
    prior=None,
    covs=None,
    n_merge=2,
    origin_cardinality=None  # <-- The dictionary from the split step
):
    """
    Simplified merge_step function.

    Steps:
    1) Identify empty clusters => clusters_to_suppress.
    2) Among non-empty clusters, for each cluster i:
       (a) compute how many neighbors it wants: origin_cardinality[i] or n_merge
       (b) propose merging i + that many neighbors
       (c) call merge_rule(...) once
       (d) if accepted, remove them from availability
    """

    mus_to_merge = []
    highest_ll_mus = []
    clusters_to_suppress = []

    print(f"\n[merge_step] Starting merge with K={K}, n_merge={n_merge}")
    print(f"raise_merge_proposals={raise_merge_proposals}")
    print('origin_cardinality :',origin_cardinality)
    

    # 1) Identify empty clusters
    cluster_counts = torch.zeros(K, dtype=torch.int64)
    cluster_assignments = logits.argmax(-1)
    for k in range(K):
        cluster_counts[k] = (cluster_assignments == k).sum()
    empty_clusters = (cluster_counts == 0).nonzero(as_tuple=False).squeeze().tolist()
    if isinstance(empty_clusters, int):
        empty_clusters = [empty_clusters]
    print(f"Empty clusters: {empty_clusters}")

    clusters_to_suppress.extend(empty_clusters)
    clusters_available = set(range(K)) - set(empty_clusters)
    if not clusters_available:
        print("[merge_step] No non-empty clusters => nothing to merge.")
        return mus_to_merge, highest_ll_mus, clusters_to_suppress

    print(f"[merge_step] clusters_available={clusters_available}")

    if raise_merge_proposals != "brute_force_NN":
        raise NotImplementedError("[merge_step] Only 'brute_force_NN' is supported in this snippet.")

    # 2) We fit a single NearestNeighbors with the maximum needed neighbors
    available_indices = sorted(list(clusters_available))
    mus_available = mus[available_indices]

    # Determine the maximum cardinality across all clusters
    max_card = 0
    if origin_cardinality is not None:
        for idx in available_indices:
            c = origin_cardinality.get(idx, n_merge)  # fallback if not found
            max_card = max(max_card, c)
    else:
        max_card = n_merge

    # We'll fit up to max(1, min(max_card-1, len(available_indices)-1)) neighbors
    global_n_neighbors = max(1, min(max_card - 1, len(available_indices) - 1))
    print('GLOBAL N NEIGHBORS : ',global_n_neighbors)
    neigh = NearestNeighbors(n_neighbors=global_n_neighbors, metric='cosine')
    neigh.fit(mus_available.cpu())
    neighbor_indices_all = neigh.kneighbors(return_distance=False)

    # row -> cluster label
    index_to_cluster = {i: cluster_idx for i, cluster_idx in enumerate(available_indices)}

    # We'll keep track of which clusters we've tried
    visited = set()

    # 3) For each cluster, propose EXACT group [i + neighbors]
    for idx_in_av, cluster_i in enumerate(available_indices):
        if cluster_i not in clusters_available:
            continue
        if cluster_i in visited:
            continue

        # how many neighbors do we want?
        if origin_cardinality is not None and cluster_i in origin_cardinality:
            c = origin_cardinality[cluster_i]
        else:
            c = n_merge
        
        
        n_neighbors_i = max(1, min(c - 1, len(available_indices) - 1))
        print('CLUSTER_i :',cluster_i)
        print('nb neighboors i :',n_neighbors_i)
        # retrieve top n_neighbors_i from neighbor_indices_all
        full_neighbors = neighbor_indices_all[idx_in_av][:n_neighbors_i]
        neighbor_clusters = [index_to_cluster[n_idx] for n_idx in full_neighbors]

        proposed_group = [cluster_i] + neighbor_clusters
        # filter out any cluster that's empty or removed
        proposed_group = [p for p in proposed_group if p in clusters_available]
        proposed_group = list(set(proposed_group))  # remove duplicates

        if len(proposed_group) < 2:
            visited.update(proposed_group)
            continue

        print(f"\n[merge_step] Proposing to merge group={proposed_group}")
        merge_decision, highest_ll = merge_rule(
            mus=mus,
            logits=logits,
            codes=codes,
            cluster_indices=proposed_group,
            alpha=alpha,
            cov_const=cov_const,
            merge_prob=merge_prob,
            prior=prior,
            covs=covs
        )

        print(f"[merge_step] Decision={merge_decision}, highest_ll={highest_ll}")
        if merge_decision[0] is True:
            # accepted => remove them
            for p in proposed_group:
                if p in clusters_available:
                    clusters_available.remove(p)

            mus_to_merge.append(proposed_group)
            highest_ll_mus.append(highest_ll)
            print(f"[merge_step] Merged group: {proposed_group}")
        else:
            print(f"[merge_step] Merge refused for {proposed_group}.")

        visited.update(proposed_group)

    print(f"[merge_step] Final merges: {mus_to_merge}")
    print(f"[merge_step] Clusters to suppress: {clusters_to_suppress}")
    return mus_to_merge, highest_ll_mus, clusters_to_suppress



def log_Hastings_ratio_merge_deterministic(alpha,
    N_s_list,
    log_ll_s_list,
    N_c,
    log_ll_merged,
    merge_prob=None,
    q_log_list=None,
    q_log_merged=None):
    """
      Computes the log Hastings ratio for the merge move, defined as the reciprocal
      of the split Hastings ratio:
      log r_merge = - log r_split
                       = -(K-1)log a
                         - sum_s log G(N_s)
                         + log G(N_c)
                         - sum_s log p(x_s,µ_s)
                         + log p(x_c,µ_c)
                         - log q(µ_c|x_c)
                         + sum_s log q(µ_s|x_s)
    """
    # Number of subclusters to be merged
    K = len(N_s_list)

    # Tensorify inputs
    alpha_t = torch.tensor(alpha, dtype=torch.float)
    N_s_t = [torch.tensor(n, dtype=torch.float) for n in N_s_list]
    N_c_t = torch.tensor(N_c, dtype=torch.float)

    # log-likelihoods
    ll_s_t = [torch.tensor(ll, dtype=torch.float) for ll in log_ll_s_list]
    ll_merged_t = torch.tensor(log_ll_merged, dtype=torch.float)

    # proposal-log terms
    #if q_log_list is None: q_log_list = [0.0]*K
    #if q_log_merged is None: q_log_merged = 0.0
    q_s_t = [torch.tensor(q, dtype=torch.float) for q in q_log_list]
    q_c_t = torch.tensor(q_log_merged, dtype=torch.float)

    # Gamma-terms
    logG_s = [torch.lgamma(n) for n in N_s_t]
    logG_c = torch.lgamma(N_c_t)

    # Build log-ratio:
    log_merge = (
        -(K - 1) * torch.log(alpha_t)
        - sum(logG_s)
        +  logG_c
        - sum(ll_s_t)
        +  ll_merged_t
        +  sum(q_s_t)
        -  q_c_t
    )

    # scalarize
    H = log_merge.item()

    # compute merge probability (if not given)
    if merge_prob is None:
        merge_prob = math.exp(H) if -700 < H < 700 else (0.0 if H < 0 else float('inf'))

    # accept-reject logic
    accept = (H > 0) or (merge_prob > torch.rand(1).item())

    # debug prints
    print(f"K={K}, H_merge={H:.4f}, merge_prob={merge_prob:.4f}, accept={accept}")

    return accept


def log_Hastings_ratio_merge(
    alpha, N_k_list, log_ll_k_list, N_c, log_ll_k_merged, merge_prob,
    q_RandomParamProposalt_list, q_RandomParamProposalt_merged
):
    """
    Computes the generalized Randomized Hastings ratio for merging multiple clusters.
    """
    print("\n[log_Hastings_ratio_merge] Calculating Hastings ratio for merge:")
    print(f"alpha: {alpha}")
    print(f"N_k_list: {N_k_list}")
    print(f"Total N_c: {N_c}")
    print(f"log_ll_k_list: {log_ll_k_list}")
    print(f"log_ll_k_merged: {log_ll_k_merged}")
    print(f"q_RandomParamProposalt_list: {q_RandomParamProposalt_list}")
    print(f"q_RandomParamProposalt_merged: {q_RandomParamProposalt_merged}")

    # Number of clusters being merged
    K = len(N_k_list)
    print(f"Number of clusters being merged K: {K}")

    # Convert lists to tensors
    N_k_tensor = torch.tensor(N_k_list, dtype=torch.float)
    log_ll_k_tensor = torch.tensor(log_ll_k_list, dtype=torch.float)
    q_RandomParamProposalt_tensor = torch.tensor(q_RandomParamProposalt_list, dtype=torch.float)

    # Convert scalars to tensors
    N_c_tensor = torch.tensor(N_c, dtype=torch.float)
    alpha_tensor = torch.tensor(alpha, dtype=torch.float)

    # Compute terms involving Gamma functions
    alpha_over_K = alpha_tensor / K
    print(f"alpha / K: {alpha_over_K}")

    # Numerator terms
    log_numerator = (
        torch.lgamma(alpha_tensor) +
        torch.lgamma(N_c_tensor) +
        torch.sum(torch.lgamma(alpha_over_K + N_k_tensor)) +
        log_ll_k_merged
    )

    # Denominator terms
    log_denominator = (
        (K - 1) * torch.log(alpha_tensor) +
        K * torch.lgamma(alpha_over_K) +
        torch.lgamma(alpha_tensor + N_c_tensor) +
        torch.sum(torch.lgamma(N_k_tensor)) +
        torch.sum(log_ll_k_tensor)
    )

    # Proposal distribution terms
    log_q_ratio = torch.sum(q_RandomParamProposalt_tensor) - q_RandomParamProposalt_merged

    # Hastings ratio
    H = log_numerator - log_denominator + log_q_ratio

    print(f"log_numerator: {log_numerator}")
    print(f"log_denominator: {log_denominator}")
    print(f"log_q_ratio: {log_q_ratio}")
    print(f"Hastings ratio H: {H}")
  
    # Convert H to a scalar if it's a tensor
    H = H.item() if isinstance(H, torch.Tensor) else H
    
    # Clamp H to prevent overflow
    #H = min(H,700) if H>0 else max(H,-700)
    #print(f"Clamped H value: {H}")
    
    if H > 0:
        exp_merge_prob = math.exp(H) if H < 700 else float('inf')  # Clamp upper bound
    else:
        exp_merge_prob = math.exp(H) if H > -700 else 0.0  # Clamp lower bound to avoid underflow
    
    merge_prob_value = merge_prob if merge_prob is not None else exp_merge_prob
    print(f"Computed merge_prob_value: {merge_prob_value}")
    
    random_value = torch.rand(1)
    print(f"Random value for decision: {random_value}")
    
    result = H > 0 or (merge_prob_value > random_value)
    
    print(f"Merge decision result: {result}")
    
    return result





def merge_rule_2sub(mus, logits, codes, k_inds, alpha, cov_const, merge_prob, prior=None,covs=None):
    """
    Gets an input a random permutation of indices of the clusters to consider merge.
    We will consider merges of pairs.
    Returns:
    (1) boolean array of size len(k_inds)//2 with the merge decision for every pair
    (2) a list of the indices of the clusterwith the highest likelihood from each pair
    """
    decisions = []
    highest_ll = []

    for i in range(0, len(k_inds), 2):
        # for each pair do
        k_1 = k_inds[i]
        if len(k_inds) - 1 == i:
            # only one cluster
            decisions.append(False)
            highest_ll.append(k_inds[i])
            return decisions, highest_ll
        k_2 = k_inds[i + 1]

        codes_ind_k1 = logits.argmax(-1) == k_1
        codes_ind_k2 = logits.argmax(-1) == k_2
        codes_ind_k = torch.logical_or(codes_ind_k1, codes_ind_k2)

        codes_k_1 = codes[codes_ind_k1]
        codes_k_2 = codes[codes_ind_k2]
        codes_k = codes[codes_ind_k]

        N_k_1 = len(codes_k_1)
        N_k_2 = len(codes_k_2)
        #N_k = N_k_1 + N_k_2
        N_k=len(codes_k)

        if N_k > 0:
            #mus_mean = (N_k_1 / N_k) * mus[k_1] + (N_k_2 / N_k) * mus[k_2]
            percentage=N_k_2/N_k
            mus_mean=rotate_vector_a_to_b(mus[k_1],mus[k_2], percentage=percentage)
            r_mean=(logits[:, k_1] + logits[:, k_2]).reshape(-1, 1)
            covs_mean = compute_data_covs_soft_assignment(
                logits=r_mean,
                codes=codes,
                K=1,
                mus=mus_mean,
                prior_name=prior.name
                )
            D=len(codes[0])
            #print("r_mean size",r_mean.size())
            if prior.prior_choice=='dynamic_data_std':
              # Compute Karcher mean of codes_k
              #karcher_mean_k = KarcherMean(None, codes_k)
              # Compute log mapping
              log_map_k = Log_mapping(codes_k, mus_mean)
              # Compute standard deviation
              std_k = log_map_k.std(axis=0)
              # Compute psi prior
              psi_mean = torch.diag(std_k) * prior.get_prior_sigma_scale()
              psi_mean = psi_mean.double()
              covs_mean = prior.compute_post_cov(r_mean.sum(axis=0), covs_mean, D, psi_index=None, psi_value=psi_mean)          
            else:        
              covs_mean=prior.compute_post_cov(r_mean.sum(axis=0), covs_mean, D)
            
        else:
            # in case both are empty clusters
            #mus_mean = torch.mean(torch.stack([mus[k_1], mus[k_2]]), axis=0)
            mus_mean=rotate_vector_a_to_b(mus[k_1],mus[k_2], percentage=0.5)
            covs_mean =covs[k_1]
        if prior is None:
            (log_ll_k, log_ll_k_1, log_ll_k_2,) = compute_split_log_ll(
                mus_mean, mus[k_1], mus[k_2], cov_const, codes_k, codes_k_1, codes_k_2
            )
        else:
            if prior.prior_choice=='dynamic_data_std':
              log_ll_k = prior.log_marginal_likelihood_dynamic_psi(codes_k, covs_mean,psi_index=None,psi_value=psi_mean)
              log_ll_k_1 = prior.log_marginal_likelihood_dynamic_psi(codes_k_1, covs[k_1],k_1)
              log_ll_k_2 = prior.log_marginal_likelihood_dynamic_psi(codes_k_2, covs[k_2],k_2)
            else :
              log_ll_k = prior.log_marginal_likelihood(codes_k, covs_mean)
              log_ll_k_1 = prior.log_marginal_likelihood(codes_k_1, covs[k_1])
              log_ll_k_2 = prior.log_marginal_likelihood(codes_k_2, covs[k_2])
              
            #DeepDPM
            #log_ll_k = prior.log_marginal_likelihood(codes_k, mus_mean)
            #log_ll_k_1 = prior.log_marginal_likelihood(codes_k_1, mus[k_1])
            #log_ll_k_2 = prior.log_marginal_likelihood(codes_k_2, mus[k_2])
        
        q_RandomParamProposalt_k_1=prior.log_mean_projected_data(codes_k_1,  mus[k_1], covs[k_1])
        q_RandomParamProposalt_k_2=prior.log_mean_projected_data(codes_k_2,  mus[k_2], covs[k_2])
        q_RandomParamProposalt_merge=prior.log_mean_projected_data(codes_k, mus_mean, covs_mean)
        decisions.append(log_Hastings_ratio_merge(alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, merge_prob,q_RandomParamProposalt_k_1,q_RandomParamProposalt_k_2,q_RandomParamProposalt_merge))
        highest_ll.append(k_inds[i: i + 2][int(log_ll_k_1 < log_ll_k_2)])
    return decisions, highest_ll
    
def merge_rule_DPM(mus, logits, codes, k_inds, alpha, cov_const, merge_prob, prior=None):
    """
    Gets an input a random permutation of indices of the clusters to consider merge.
    We will consider merges of pairs.
    Returns:
    (1) boolean array of size len(k_inds)//2 with the merge decision for every pair
    (2) a list of the indices of the clusterwith the highest likelihood from each pair
    """
    decisions = []
    highest_ll = []

    for i in range(0, len(k_inds), 2):
        # for each pair do
        k_1 = k_inds[i]
        if len(k_inds) - 1 == i:
            # only one cluster
            decisions.append(False)
            highest_ll.append(k_inds[i])
            return decisions, highest_ll
        k_2 = k_inds[i + 1]

        codes_ind_k1 = logits.argmax(-1) == k_1
        codes_ind_k2 = logits.argmax(-1) == k_2
        codes_ind_k = torch.logical_or(codes_ind_k1, codes_ind_k2)

        codes_k_1 = codes[codes_ind_k1]
        codes_k_2 = codes[codes_ind_k2]
        codes_k = codes[codes_ind_k]

        N_k_1 = len(codes_k_1)
        N_k_2 = len(codes_k_2)
        N_k = N_k_1 + N_k_2

        if N_k > 0:
            mus_mean = (N_k_1 / N_k) * mus[k_1] + (N_k_2 / N_k) * mus[k_2]
        else:
            # in case both are empty clusters
            mus_mean = torch.mean(torch.stack([mus[k_1], mus[k_2]]), axis=0)
        if prior is None:
            (log_ll_k, log_ll_k_1, log_ll_k_2,) = compute_split_log_ll(
                mus_mean, mus[k_1], mus[k_2], cov_const, codes_k, codes_k_1, codes_k_2
            )
        else:
            log_ll_k = prior.log_marginal_likelihood(codes_k, mus_mean)
            log_ll_k_1 = prior.log_marginal_likelihood(codes_k_1, mus[k_1])
            log_ll_k_2 = prior.log_marginal_likelihood(codes_k_2, mus[k_2])

        decisions.append(log_Hastings_ratio_merge(alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, merge_prob))
        highest_ll.append(k_inds[i: i + 2][int(log_ll_k_1 < log_ll_k_2)])
    return decisions, highest_ll
