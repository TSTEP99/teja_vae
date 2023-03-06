"""File to test the functionality of all implemented modules in Teja-VAE"""
from teja_encoder import teja_encoder
from teja_decoder import teja_decoder
from teja_vae import teja_vae
import torch

def test_encoder():
    """Function to test the teja encoder module"""

    sample_tensor = torch.randn(14052, 19, 45)
    other_dims = sample_tensor.shape[1:]
    encoder = teja_encoder(other_dims = other_dims)
    means, log_vars = encoder(sample_tensor)

    assert tuple(means.shape) == (14052, 3)
    assert tuple(log_vars.shape) == (14052, 3)

def test_decoder():
    """Function to test the teja decoder module"""

    means = torch.randn((14052,3))
    log_vars = torch.randn((14052,3))
    other_dims = [19, 45]

    decoder = teja_decoder(other_dims = other_dims)
    reconstructed_tensor =decoder(means, log_vars)

    assert tuple(reconstructed_tensor.shape) == (14052, 19, 45)

def test_vae():
    """Function to test Teja-VAE"""

    device = "cuda:0"
    sample_tensor = torch.randn(14052, 19, 45, device = device)
    other_dims = sample_tensor.shape[1:]
    vae = teja_vae(other_dims = other_dims, device = device)
    reconstructed_tensor = vae(sample_tensor)

    assert sample_tensor.shape == reconstructed_tensor.shape

if __name__ == "__main__":
    
    test_encoder()
    test_decoder()
    test_vae()