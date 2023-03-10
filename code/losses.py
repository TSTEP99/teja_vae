"""File containing the losses for Teja-VAE"""
import torch

def reconstruction_loss(samples, mean, log_var):
    """Computes the reconstruction loss in similair fashion to VAE CP"""
    #print(mean.shape, log_var.shape, samples.shape)
    L = samples.shape[0]
    std = torch.sqrt(torch.exp(log_var))

    return (-torch.log(std) - 0.5 * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * ((samples - mean)/std) ** 2 ).sum()/L

def regularization_loss(mus, lambdas, mus_tildes, lambdas_tildes):
    """Computes the regularization loss in a similair fashion to the VAE-CP paper"""
    loss = 0
    for i in range(len(mus)):
        mu = mus[i]
        lambda_ = lambdas[i]
        mu_tilde = mus_tildes[i]
        lambda_tilde = lambdas_tildes[i]

        var = torch.exp(lambda_)
        var_tilde = torch.exp(lambda_tilde)
        std = torch.sqrt(var)
        std_tilde = torch.sqrt(var_tilde)
        loss += (torch.log(std_tilde/std) + (var + (mu - mu_tilde)**2)/(2 * var_tilde) - 0.5).sum()
    return loss

def original_loss(samples, means, log_vars, mus, lambdas, mus_tildes, lambdas_tildes):
    """Computes a loss function very similair to the VAE-CP paper"""
    rec_loss = -reconstruction_loss(samples, means, log_vars)
    reg_loss = regularization_loss(mus, lambdas, mus_tildes, lambdas_tildes)
    #print(rec_loss, reg_loss)
    return rec_loss + reg_loss