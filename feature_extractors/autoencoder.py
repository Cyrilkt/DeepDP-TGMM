#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import torch
import torch.nn as nn
from collections import OrderedDict
import math


"""
class AutoEncoder(nn.Module):
    def __init__(self, args, input_dim):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.latent_dim = args.latent_dim
        self.hidden_dims = args.hidden_dims
        self.hidden_dims.append(self.latent_dim)
        self.dims_list = (
            args.hidden_dims + args.hidden_dims[:-1][::-1]
        )  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.n_clusters = args.n_clusters

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == args.latent_dim

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                # Use ReLU for the first layer unless it's the only one, in which case use Tanh.
                activation = nn.ReLU() if len(self.hidden_dims) > 1 else nn.Tanh()
                layers.update(
                    {
                        "linear0": nn.Linear(self.input_dim, hidden_dim),
                        "activation0": activation,
                    }
                )
            else:
                if idx == len(self.hidden_dims) - 1:
                    # Final encoder layer: use Tanh to map outputs to [-1, 1]
                    layers.update(
                        {
                            "linear{}".format(idx): nn.Linear(self.hidden_dims[idx - 1], hidden_dim),
                            "activation{}".format(idx): nn.Tanh(),
                            "bn{}".format(idx): nn.BatchNorm1d(hidden_dim),
                        }
                    )
                else:
                    layers.update(
                        {
                            "linear{}".format(idx): nn.Linear(self.hidden_dims[idx - 1], hidden_dim),
                            #"activation{}".format(idx): nn.Tanh(),
                            "bn{}".format(idx): nn.BatchNorm1d(hidden_dim),
                        }
                    )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                # Final decoder layer: add Tanh to output values in [-1, 1]
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(hidden_dim, self.output_dim),
                        #"output_activation": nn.ReLU(),
                    }
                )
            else:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(hidden_dim, tmp_hidden_dims[idx + 1]),
                        "activation{}".format(idx): nn.ReLU(),
                        "bn{}".format(idx): nn.BatchNorm1d(tmp_hidden_dims[idx + 1]),
                    }
                )
        self.decoder = nn.Sequential(layers)

    def forward(self, X, latent=False):
        # Pass through the encoder
        z = self.encoder(X)
        # Normalize latent vector so it lies on the unit hypersphere
        z = F.normalize(z, p=2, dim=1)
        if latent:
            return z
        return self.decoder(z)
"""
class AutoEncoder_adecoch(nn.Module):
    def __init__(self, args, input_dim):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.latent_dim = args.latent_dim
        self.hidden_dims = args.hidden_dims
        self.hidden_dims.append(self.latent_dim)
        self.dims_list = (
            args.hidden_dims + args.hidden_dims[:-1][::-1]
        )  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.n_clusters = args.n_clusters

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == args.latent_dim

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                # Use ReLU for the first layer unless it's the only one, in which case use Tanh.
                activation = nn.ReLU() if len(self.hidden_dims) > 1 else nn.Tanh()
                layers.update(
                    {
                        "linear0": nn.Linear(self.input_dim, hidden_dim),
                        "activation0": activation,
                    }
                )
            else:
                if idx == len(self.hidden_dims) - 1:
                    # Final encoder layer: use Tanh to map outputs to [-1, 1]
                    layers.update(
                        {
                            "linear{}".format(idx): nn.Linear(self.hidden_dims[idx - 1], hidden_dim),
                            #"bn{}".format(idx): nn.BatchNorm1d(hidden_dim),
                            #"ln{}".format(idx): nn.LayerNorm(hidden_dim),
                            "activation{}".format(idx): nn.Tanh(),
                            #"activation{}".format(idx): nn.ReLU(),
                            #"bn{}".format(idx): nn.BatchNorm1d(hidden_dim),
                            #"ln{}".format(idx): nn.LayerNorm(hidden_dim),
                        }
                    )
                else:
                    layers.update(
                        {
                            "linear{}".format(idx): nn.Linear(self.hidden_dims[idx - 1], hidden_dim),
                            "activation{}".format(idx): nn.ReLU(),
                            "bn{}".format(idx): nn.BatchNorm1d(hidden_dim),
                        }
                    )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                # Final decoder layer: add Tanh to output values in [-1, 1]
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(hidden_dim, self.output_dim),
                        #"output_activation": nn.ReLU(),
                    }
                )
            else:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(hidden_dim, tmp_hidden_dims[idx + 1]),
                        "activation{}".format(idx): nn.ReLU(),
                        "bn{}".format(idx): nn.BatchNorm1d(tmp_hidden_dims[idx + 1]),
                    }
                )
        self.decoder = nn.Sequential(layers)

    def forward(self, X, latent=False):
        # Pass through the encoder
        z = self.encoder(X)
        # Normalize latent vector so it lies on the unit hypersphere
        z = F.normalize(z, p=2, dim=1)
        if latent:
            return z
        return self.decoder(z)

class AutoEncoder(nn.Module):
    def __init__(self, args, input_dim):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.latent_dim = args.latent_dim
        self.hidden_dims = args.hidden_dims
        self.hidden_dims.append(self.latent_dim)
        self.dims_list = (
            args.hidden_dims + args.hidden_dims[:-1][::-1]
        )  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.n_clusters = args.n_clusters

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == args.latent_dim

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        "linear0": nn.Linear(self.input_dim, hidden_dim),
                        #"bn{}".format(idx): nn.BatchNorm1d(hidden_dim),
                        #"ln{}".format(idx): nn.LayerNorm(hidden_dim),
                        #"rmsn{}".format(idx) : nn.RMSNorm(hidden_dim),
                        #"activation0": nn.ReLU(),
                        #"ln{}".format(idx): nn.LayerNorm(hidden_dim),
                        "activationN": nn.Tanh(),
                        #"bn{}".format(idx): nn.BatchNorm1d(hidden_dim),
                        #"activation0": nn.LeakyReLU(),
                    }
                )
            else:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(
                            self.hidden_dims[idx - 1], hidden_dim
                        ),
                        "activation{}".format(idx): nn.ReLU(),
                        "bn{}".format(idx): nn.BatchNorm1d(self.hidden_dims[idx]),
                    }
                )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(hidden_dim, self.output_dim),
                        #"activationN": nn.Tanh(),
                    }
                )
            else:
                layers.update(
                    {
                        "linear{}".format(idx): nn.Linear(
                            hidden_dim, tmp_hidden_dims[idx + 1]
                        ),
                        "activation{}".format(idx): nn.ReLU(),
                        "bn{}".format(idx): nn.BatchNorm1d(tmp_hidden_dims[idx + 1]),
                    }
                )
        self.decoder = nn.Sequential(layers)

    def __repr__(self):
        repr_str = "[Structure]: {}-".format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
            repr_str += "{}-".format(dim)
        repr_str += str(self.output_dim) + "\n"
        repr_str += "[n_layers]: {}".format(self.n_layers) + "\n"
        repr_str += "[n_clusters]: {}".format(self.n_clusters) + "\n"
        repr_str += "[input_dims]: {}".format(self.input_dim)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        output = self.encoder(X)
        z = F.normalize(z, p=2, dim=1)
        if latent:
            return output
        return self.decoder(output)

    def decode(self, latent_X):
        return self.decoder(latent_X)



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# New Encoder: For input shape [batch, 1, 32, 32]
class Encoder(nn.Module):
    def __init__(self, channels, image_size, embedding_dim):
        super(Encoder, self).__init__()
        # Using three convolutional blocks:
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(32),  # Batch Normalization
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # NEW BLOCK
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Save shape before flattening for later reshaping in the decoder.
        # New shape: (256, image_size//16, image_size//16)
        self.shape_before_flattening = (256, image_size // 16, image_size // 16)
        self.flatten = nn.Flatten()
        # For image_size=32: 256*(32//16)*(32//16) = 256*2*2 = 1024
        self.fc = nn.Linear(256 * (image_size // 16) * (image_size // 16), embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x_flat = self.flatten(x)
        embedding = self.fc(x_flat)
        # Apply tanh activation then L2 normalization on the latent representation.
        embedding = torch.tanh(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

# New Decoder: Upsamples from latent space back to [batch, 1, 32, 32]
class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening, channels):
        super(Decoder, self).__init__()
        self.shape_before_flattening = shape_before_flattening  # e.g., (128,4,4)
        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Outputs should be in range [0,1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), *self.shape_before_flattening)
        x = self.decoder(x)
        return x

# Full autoencoder that preserves method names from your previous implementation
class ConvAutoEncoder(nn.Module):
    def __init__(self, args, input_dim):
        """
        Args:
            args: must have an attribute 'latent_dim'
            input_dim: e.g., 784 for a 28x28 image (before padding).
                       The transform should pad to 32x32.
        """
        super(ConvAutoEncoder, self).__init__()
        self.args = args
        self.input_dim = input_dim      # e.g., 784
        self.output_dim = self.input_dim
        self.latent_dim = args.latent_dim

        # For FashionMNIST: original images are 28x28; after transform (Pad(2)) they become 32x32.
        image_size = 32  
        channels = 1

        # Use the new Encoder and Decoder definitions
        self.encoder = Encoder(channels, image_size, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.encoder.shape_before_flattening, channels)

    def encode(self, x):
        """
        Encode input image (or flattened image) to latent representation.
        """
        # If input is flattened ([batch, input_dim]), reshape to [batch, 1, sqrt, sqrt].
        was_flat = False
        if x.dim() == 2 and x.size(1) == self.input_dim:
            was_flat = True
            sqrt_dim = int(self.input_dim ** 0.5)
            x = x.view(x.size(0), 1, sqrt_dim, sqrt_dim)
        latent_vec = self.encoder(x)
        return latent_vec

    def decode(self, latent_vec):
        """
        Decode latent vector back to image.
        """
        x = self.decoder(latent_vec)
        return x

    def forward(self, x, latent=False):
        """
        If latent is True, returns the latent vector.
        Otherwise, returns the reconstructed image.
        If input was flattened, output is flattened as well.
        """
        was_flat = False
        if x.dim() == 2 and x.size(1) == self.input_dim:
            was_flat = True
            sqrt_dim = int(self.input_dim ** 0.5)
            x = x.view(x.size(0), 1, sqrt_dim, sqrt_dim)
        
        latent_vec = self.encode(x)
        if latent:
            return latent_vec
        
        decoded = self.decode(latent_vec)
        if was_flat:
            decoded = decoded.view(decoded.size(0), -1)
        return decoded

# If you need the Flatten and UnFlatten classes for other parts of your project, you can keep them as is.
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1).float()

class UnFlatten(nn.Module):
    def __init__(self, channel, width):
        super().__init__()
        self.channel = channel
        self.width = width

    def forward(self, x):
        return x.reshape(-1, self.channel, self.width, self.width)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoEncoder_v2(nn.Module):
    def __init__(self, args, input_dim):
        super(ConvAutoEncoder, self).__init__()
        self.args = args
        self.input_dim = input_dim       # e.g., 784 for a 28x28 image flattened
        self.output_dim = self.input_dim
        self.latent_dim = args.latent_dim
        
        # Encoder: expects input shape [batch, 1, sqrt(input_dim), sqrt(input_dim)]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # Output: [batch, 32, sqrt/2, sqrt/2]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: [batch, 64, sqrt/4, sqrt/4]
            nn.BatchNorm2d(64),
            #nn.LeakyReLU()
        )
        
        # Fully connected layers for latent space transformation
        # For a 28x28 input, sqrt(input_dim)==28, so after two conv layers the feature map is 7x7.
        self.fc_enc = nn.Linear(64 * 7 * 7, self.latent_dim)
        self.fc_dec = nn.Linear(self.latent_dim, 64 * 7 * 7)
        
        # Decoder: upsamples back to image shape [batch, 1, 28, 28]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # Output: [batch, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: [batch, 32, 28, 28]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),   # Output: [batch, 1, 28, 28]
            nn.Tanh()
        )
    
    def encode(self, x):
        # Encode image to latent representation
        print('BONJOUR :',x.size())
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        latent_vec = self.fc_enc(x)
        # Apply L2 normalization on the latent representation
        latent_vec = F.normalize(latent_vec, p=2, dim=1)
        return latent_vec
    
    def decode(self, latent_vec):
        # Decode latent vector back into image
        x = self.fc_dec(latent_vec)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.decoder(x)
        return x
    
    def forward(self, x, latent=False):
        # Check if input is flattened (i.e., shape [batch, nb_dim]).
        # If so, reshape to [batch, 1, sqrt(nb_dim), sqrt(nb_dim)].
        was_flat = False
        print('BJRR')
        if x.dim() == 2 and x.size(1) == self.input_dim:
            was_flat = True
            sqrt_dim = int(self.input_dim ** 0.5)
            x = x.view(x.size(0), 1, sqrt_dim, sqrt_dim)
        
        latent_vec = self.encode(x)
        if latent:
            return latent_vec
        
        decoded = self.decode(latent_vec)
        # If the original input was flattened, re-flatten the output.
        if was_flat:
            decoded = decoded.view(decoded.size(0), -1)
        return decoded




        
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1).float()


class UnFlatten(torch.nn.Module):

    def __init__(self, channel, width) -> None:
        super().__init__()
        self.channel = channel
        self.width = width

    def forward(self, x):
        return x.reshape(-1, self.channel, self.width, self.width)


class ConvAutoEncoder_deepdpm(nn.Module):
    def __init__(self, args, input_dim):
        super(ConvAutoEncoder, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.latent_dim = args.latent_dim

        # encoder #
        self.encoder_conv = nn.Sequential(
            UnFlatten(channel=1, width=16),                       # [batch, 1, 16, 16]
            nn.Conv2d(1, 32, 5, stride=1),                       # [batch, 32, 12, 12]
            nn.BatchNorm2d(32),                                  # [batch, 32, 12, 12]
            nn.ReLU(),

        )
        self.encoder_maxPool = nn.MaxPool2d(2, stride=2, return_indices=True)  # [batch, 32, 6, 6]
        self.encoder_linear = nn.Sequential(
            Flatten(),                                           # [batch, 1152]
            nn.Linear(32 * 6 * 6, self.latent_dim)
        )

        # decoder #
        self.decoder_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 32 * 6 * 6),
            UnFlatten(channel=32, width=6),
        )                                                       # [batch, 32, 6, 6]
        self.decoder_maxPool = nn.MaxUnpool2d(2, stride=2)      # [batch, 32, 12, 12]
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, 5, stride=1),              # [batch, 1, 16, 16]
            Flatten()
        )

    def forward(self, X, latent=False):
        output = self.encode(X)
        if latent:
            return output
        return self.decode(output)

    def encode(self, X):
        out = self.encoder_conv(X)
        out, self.ind = self.encoder_maxPool(out)
        return self.encoder_linear(out)

    def decode(self, X):
        out = self.decoder_linear(X)
        out = self.decoder_maxPool(out, self.ind)
        return self.decoder_conv(out)

