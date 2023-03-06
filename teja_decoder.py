"""Implementation for the decoder portion of Teja-VAE"""

from math import floor
import torch
import torch.nn as nn

class teja_decoder(nn.Module):
    def __init__(self, other_dims,  hidden_layer_size = 100, rank = 3, device = None):
        """Initializes the parameters and layers for the neural network for Teja-VAE"""

        #Calls constructor of super class
        super(teja_decoder, self).__init__()

        #defines the device for the module
        if device:
            self.device = device
        else:
            self.device = "cpu"

        #Initializes the prior mu and sigma for epoch/sample dimension
        self.original_mu = torch.rand((1,rank))
        self.original_lambda = torch.rand((1,rank))

        #Initializes other factor matrices used in the encoder portion
        other_mus=[]
        other_lambdas=[]
        other_mus_tildes=[]
        other_lambdas_tildes=[]

        for dim in other_dims:
            other_mus.append(nn.Parameter(torch.randn((dim, rank), requires_grad=True, device = device))) 
            other_lambdas.append(nn.Parameter(torch.randn((dim,rank), requires_grad=True, device = device))) 
            other_mus_tildes.append(nn.Parameter(torch.randn((dim, rank), requires_grad=True, device = device)))
            other_lambdas_tildes.append(nn.Parameter(torch.randn((dim, rank), requires_grad=True, device = device)))

        self.other_mus = nn.ParameterList(other_mus)
        self.other_lambdas = nn.ParameterList(other_lambdas)

        # Layers for computing the individual tensor elements
        self.FC_input = nn.Linear((len(other_dims)+1) * rank, hidden_layer_size, device = device)
        self.FC_mean = nn.Linear(hidden_layer_size, 1, device = device)
        self.FC_log_var = nn.Linear(hidden_layer_size, 1, device = device)

        #Activation function to compute the hidden layer when decoding
        self.tanh = nn.Tanh()    
        self.relu = nn.ReLU() 

        #Saves the size(s) of the non-epoch/non-sample dimensions
        self._other_dims = other_dims
    
    def forward(self, mean, log_var):
        """Computes the forward pass for Teja-VAE takes sample of Teja-VAE encoder as input"""

        dims = []
        dims.append(mean.shape[0])
        dims.extend(self._other_dims)

        indices = self._create_indices(dims)

        num_dims = indices.shape[1]

        #Makes array(s) with all the factor means and log variances
        factor_means = []
        factor_log_vars = []

        factor_means.append(mean)
        factor_log_vars.append(log_var)

        factor_means.extend(self.other_mus)
        factor_log_vars.extend(self.other_lambdas)

        #creates tensor to form u vector(s) from VAE-CP paper
        Us = []

        #Samples to create aforementioned u vector
        for i in range(num_dims):
            Us.append(self._reparameterization(factor_means[i][indices[:,i]], factor_log_vars[i][indices[:,i]]))

        #Concatenates tensors to form u vector from paper 
        U_vecs = torch.concat(Us, dim=1)

        #Pass u vector through the decoder layers to generate mean and log_var value for each tensor element
        #Note: The original paper uses the tanh function for the hidden layer
        hidden = self.tanh(self.FC_input(U_vecs))


        #NOTE: use ReLU Activation
        elements_mean = self.relu(self.FC_mean(hidden))

        elements_log_var = self.FC_log_var(hidden)

        #Samples on a per element basis using output of decoder layers
        sample_elements = self._reparameterization(elements_mean, elements_log_var)

        #Creates original shape of tensor
        original_shape=[mean.shape[0]]
        original_shape.extend(self._other_dims)

        reconstructed_tensor = sample_elements.view(*original_shape)

        return reconstructed_tensor
        

    def _create_indices(self,dims):
        """
        Takes tensor shape as input and 
        creates list of all possible indices
        """
        
        indices = []

        for dim in dims:
            indices.append(torch.arange(dim))
        indices = torch.cartesian_prod(*indices)
        return indices.long() 
    
    def _reparameterization(self, mean, log_var):
        """Uses the parameterization trick from the original VAE formulation"""

        epsilons = torch.randn_like(log_var, device = self.device)

        return mean + epsilons * torch.sqrt(torch.exp(log_var))
