# DeepDP-TGMM: Amortized Non-Parametric Clustering for Hyperspherical Self-Supervised Representations

Official code repository for the CVPR 2026 Findings paper:

**DeepDP-TGMM: Amortized Non-Parametric Clustering for Hyperspherical Self-Supervised Representations**  
Cyril Kana Tepakbong, Kévin Bouchard, Julien Maitre

## Overview

DeepDP-TGMM is a plug-and-play Bayesian non-parametric clustering framework designed for hyperspherical self-supervised representations. The method extends DP-TGMM with amortized inference: a clusternet predicts soft responsibilities in tangent spaces of the sphere, cluster-specific subclusternets model local binary refinements, and geometry-aware split-merge moves adapt the number of clusters automatically.

The method is evaluated on fixed self-supervised embeddings produced by both:

- **contrastive SSL encoders**: MoCo v1, MoCo v3
- **non-contrastive SSL encoders**: BYOL, SimSiam

across balanced and imbalanced variants of CIFAR-10, CIFAR-20, ImageNet-10, and ImageNet-100.

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/DeepDP-TGMM.git
cd DeepDP-TGMM
pip install -r requirements.txt
```

The dependencies follow the same environment as the public DeepDPM implementation.

## Data format

DeepDP-TGMM clusters precomputed embeddings. Each dataset directory is expected to contain at least:

- `train_data.pt`
- `test_data.pt`

If labels are available for evaluation, they can also be stored alongside the tensors and used with `--use_labels_for_eval`.

A typical directory layout is:

```text
pretrained_embeddings/
├── MOCO/
│   ├── cifar10/
│   ├── cifar10imbv2/
│   ├── cifar20/
│   └── cifar20imbv3/
├── byol/
│   ├── cifar10_v3_nonnormalized/
│   └── imb_version/imb_versions/cifar10_v3_nonnormalized_v2/
└── simsiam/
    ├── cifar20/
    ├── cifar20imbv3/
    ├── imagenet10/
    └── imagenet10imbv2/
```

## Representative commands

Below are a few representative commands used in the paper. They are intentionally limited to the main settings rather than reproducing every experimental run.

### 1) MoCo v1 on CIFAR-10

```bash
python DeepDPM.py \
  --dataset reuters10k \
  --lr 0.002 \
  --init_k 3 \
  --train_cluster_net 25 \
  --transform_input_data None \
  --log_metrics_at_train True \
  --dir ./pretrained_embeddings/MOCO/cifar10 \
  --offline \
  --prior_kappa 0.005 \
  --alpha 10 \
  --batch-size 256 \
  --use_labels_for_eval \
  --save_checkpoints false \
  --prior_sigma_scale 15 \
  --gpus 1 \
  --prior_sigma_choice isotropic \
  --n_sub 2 \
  --freeze_mus_submus_after_splitmerge 2 \
  --compute_params_every 1 \
  --n_merge 4 \
  --NIW_prior_nu 65 \
  --max_epochs 200 \
  --how_to_init_mu kmeans \
  --how_to_init_mu_sub umap \
  --split_merge_every_n_epochs 20
```

### 2) BYOL on CIFAR-10

```bash
python DeepDPM.py \
  --dataset reuters10k \
  --lr 0.002 \
  --init_k 3 \
  --train_cluster_net 25 \
  --transform_input_data None \
  --log_metrics_at_train True \
  --dir ./pretrained_embeddings/byol/cifar10_v3_nonnormalized \
  --offline \
  --prior_kappa 0.005 \
  --alpha 10 \
  --batch-size 256 \
  --use_labels_for_eval \
  --save_checkpoints false \
  --prior_sigma_scale 0.01 \
  --gpus 1 \
  --prior_sigma_choice isotropic \
  --n_sub 2 \
  --freeze_mus_submus_after_splitmerge 2 \
  --compute_params_every 1 \
  --n_merge 4 \
  --NIW_prior_nu 129 \
  --max_epochs 200 \
  --how_to_init_mu kmeans \
  --how_to_init_mu_sub kmeans \
  --split_merge_every_n_epochs 20
```

### 3) SimSiam on CIFAR-20 (imbalanced)

```bash
python DeepDPM.py \
  --dataset reuters10k \
  --lr 0.002 \
  --init_k 3 \
  --train_cluster_net 25 \
  --transform_input_data None \
  --log_metrics_at_train True \
  --dir ./pretrained_embeddings/simsiam/cifar20imbv3 \
  --offline \
  --prior_kappa 0.005 \
  --alpha 10 \
  --batch-size 256 \
  --use_labels_for_eval \
  --save_checkpoints false \
  --prior_sigma_scale 0.01 \
  --gpus 1 \
  --prior_sigma_choice isotropic \
  --n_sub 2 \
  --freeze_mus_submus_after_splitmerge 2 \
  --compute_params_every 1 \
  --n_merge 4 \
  --NIW_prior_nu 140 \
  --max_epochs 200 \
  --how_to_init_mu kmeans \
  --how_to_init_mu_sub kmeans \
  --split_merge_every_n_epochs 20
```

### 4) MoCo v3 on ImageNet-10

```bash
python DeepDPM.py \
  --dataset reuters10k \
  --lr 0.002 \
  --init_k 3 \
  --train_cluster_net 25 \
  --transform_input_data None \
  --log_metrics_at_train True \
  --dir ./pretrained_embeddings/mocov3/mocov3_version_papier_p2/imagenet10 \
  --offline \
  --prior_kappa 0.005 \
  --alpha 10 \
  --batch-size 128 \
  --use_labels_for_eval \
  --save_checkpoints false \
  --prior_sigma_scale 0.3 \
  --gpus 1 \
  --prior_sigma_choice isotropic \
  --n_sub 2 \
  --freeze_mus_submus_after_splitmerge 2 \
  --compute_params_every 1 \
  --n_merge 4 \
  --NIW_prior_nu 129 \
  --max_epochs 200 \
  --how_to_init_mu kmeans \
  --how_to_init_mu_sub kmeans \
  --split_merge_every_n_epochs 20
```

## Notes

- When using your own embeddings, keep the tensor format consistent with the expected `.pt` files.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{KanaTepakbong2026DeepDPTGMM,
  title={DeepDP-TGMM: Amortized Non-Parametric Clustering for Hyperspherical Self-Supervised Representations},
  author={Kana Tepakbong, Cyril and Bouchard, K{\'e}vin and Maitre, Julien},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year={2026}
}
```

## Acknowledgments

This implementation builds on the design philosophy of DeepDPM and extends it to spherical non-parametric clustering with tangent-space Gaussian modeling and amortized inference.
