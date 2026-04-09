#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

from src.feature_extractors.autoencoder import AutoEncoder, ConvAutoEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class FeatureExtractor(nn.Module):
    def __init__(self, args, input_dim):
        super(FeatureExtractor, self).__init__()
        self.args = args
        self.feature_extractor = None
        self.autoencoder = None
        self.latent_dim = None
        self.input_dim = input_dim
        #if self.args.dataset == "usps":
        #    self.autoencoder = ConvAutoEncoder(args, input_dim=self.input_dim)
        if self.args.dataset=='fashionmnist':
             print('FASHION CONV')
             self.autoencoder = ConvAutoEncoder(args, input_dim=self.input_dim)
        else:
            self.autoencoder = AutoEncoder(args, input_dim=self.input_dim)
        self.latent_dim = self.autoencoder.latent_dim

    
    def add_gaussian_noise(self, inputs, mean=0.0, std=0.1):
        # Generate Gaussian noise with the same shape as inputs
        noise = torch.normal(mean=mean, std=std, size=inputs.shape)
        noise = noise.to(inputs.device)  # Match device with inputs
        noisy_inputs = inputs + noise  # Add noise to original inputs
        return noisy_inputs
        
    def forward(self, X, latent=False,infer=False):
        if self.feature_extractor:
            X = self.feature_extractor(X)
        if self.autoencoder:
            #X=F.normalize(X,p=2,dim=1) j normalisais l'entrée et non l'espace latent ! 
            if self.args.dataset=='fashionmnist':
              if X.dim() == 2 and X.size(1) == self.input_dim:
                was_flat = True
                sqrt_dim = int(self.input_dim ** 0.5)
                X = X.view(X.size(0), 1, sqrt_dim, sqrt_dim)
              output = self.autoencoder.encoder(X)
              output=F.normalize(output,p=2,dim=1)            
              if latent:
                  output = output.view(output.size(0), -1)
                  return output
              # If the original input was flattened, re-flatten the output.
              decoded=self.autoencoder.decoder(output)
              if was_flat:
                  decoded = decoded.view(decoded.size(0), -1)
              return decoded
            else:
              output = self.autoencoder.encoder(X)
              #if not infer:
              #output=self.add_gaussian_noise(output,std=0.05)
              output=F.normalize(output,p=2,dim=1)            
              if latent:
                  return output
              # If the original input was flattened, re-flatten the output.
              decoded=self.autoencoder.decoder(output)
              return decoded
        return X
        
    def forward_v1(self, X, latent=False):
        if self.feature_extractor:
            X = self.feature_extractor(X)
        if self.autoencoder:
            output = self.autoencoder.encoder(X)
            output=F.normalize(output,p=2,dim=1)            
            if latent:
                return output
            return self.autoencoder.decoder(output)
        return X
        
    def forward_deepDPM(self, X, latent=False):
        if self.feature_extractor:
            X = self.feature_extractor(X)
        if self.autoencoder:
            output = self.autoencoder.encoder(X)
            if latent:
                return output
            return self.autoencoder.decoder(output)
        return X

    def decode(self, latent_X):
        return self.autoencoder.decoder(latent_X)

    def extract_features(self, x):
        return self.feature_extractor(x)
    
    def get_fe_model(self, output_dim=128):
        backbone = self._get_backbone()
        model = ContrastiveModel(backbone=backbone, features_dim=output_dim)

        # Load pretrained weights
        if self.args.pretrain_path is not None and os.path.exists(self.args.pretrain_path):
            state = torch.load(self.args.pretrain_path, map_location='cpu')
            model.load_state_dict(state, strict=False)
            print("Loaded pretrained weights")
        return model

    def _get_backbone(self):
        if self.args.dataset in ('cifar-10', 'cifar-20'):
            from src.feature_extractors.resnet_cifar import resnet18
            backbone = resnet18()
        elif self.args.dataset == 'stl-10':
            from src.feature_extractors.resnet_stl import resnet18
            backbone = resnet18()
        elif 'imagenet' in self.args.dataset:
            from src.feature_extractors.resnet import resnet50
            backbone = resnet50()
        return backbone

class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.features_dim = features_dim
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim = 1)
        return features
