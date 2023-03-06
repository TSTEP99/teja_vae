"""
Implementation of a VAE based CP decomposition where we used the values along the spatial dimension 
to encoder epoch/sample values of our factor matrixs while other values are calculated in the same
way as https://arxiv.org/pdf/1611.00866.pdf 
"""
from teja_encoder import teja_encoder
from teja_decoder import teja_decoder
import torch
import torch.nn as nn

class teja_vae(nn.Module):
    def __init__(self, other_dims, output_channels = 32, kernel_size = 19, stride = 1, encoder_hidden_layer_size = 100, decoder_hidden_layer_size = 100, rank = 3, device = None):
        """Initializes the parameters and layer for Teja VAE"""

        #Calls constructor of super class
        super(teja_vae, self).__init__()

        #Initialize Encoder
        self.encoder = teja_encoder(other_dims, output_channels, kernel_size, stride, encoder_hidden_layer_size, rank, device).to(device = device)

        #Initialize Decoder
        self.decoder = teja_decoder(other_dims, decoder_hidden_layer_size, rank, device).to(device = device)


    def forward(self, x):
        """Does the forward pass for Teja VAE"""

        #Passes x through encoder of VAE
        means, log_vars = self.encoder(x)

        #Passes mean and log variance throught decoder to try to reconstruct original tensor
        x_hat = self.decoder(means, log_vars)

        return x_hat