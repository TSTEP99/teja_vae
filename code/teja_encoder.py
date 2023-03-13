"""Implementation for the encoder portion of Teja-VAE"""

from math import floor
import torch
import torch.nn as nn

class teja_encoder(nn.Module):
    def __init__(self, other_dims, output_channels = 32, kernel_size = 19, stride = 1, hidden_layer_size = 100, rank = 3, device = None):
        """Initializes the parameters and layers for encoder portion of Teja-VAE"""
        
        #Calls constructor of super class
        super(teja_encoder, self).__init__()

        #defines the device for the module
        if device:
            self.device = device
        else:
            self.device = "cpu"

        #Rewrite kernel_size as tuple if int
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        #Rewrite stride as tuple if int
        if isinstance(stride, int):
            stride = (stride, stride)

        # #First (and possibly only) convolutional layer
        # self.conv1 = nn.Conv2d(1, output_channels, kernel_size = kernel_size, stride = stride, device = device)

        # #Computed input size (input_size) to fully connected layer (formulas are from pytorch documentation)
        # h_out = floor((other_dims[0] - (kernel_size[0] - 1) - 1)/stride[0] + 1)
        # w_out = floor((other_dims[1] - (kernel_size[1] - 1) - 1)/stride[1] + 1)
        # input_size = h_out * w_out * output_channels

        #Fully Connected Layer to compute hidden layer
        self.FC_input = nn.Linear(in_features = torch.prod(torch.tensor(other_dims)).item(), out_features = hidden_layer_size, device = device)

        #Fully Connected Layer to compute mean of latent space
        self.FC_mean = nn.Linear(in_features = hidden_layer_size, out_features = rank, device = device)

        #Fully Connected Layer to computer log variance of latent space
        self.FC_log_var = nn.Linear(in_features = hidden_layer_size, out_features = rank, device = device)
        
        #Activation function to use in hidden layer computation
        self.activation = torch.nn.ReLU()


        
    def forward(self, x):
        """Forward operation of Teja-VAE computes the mean and log variance of the epoch/sample matrices"""
        
        # #Add a channel dimension in the original dimensions of the tensor
        # if x.shape[0] != 4:
        #    x = torch.unsqueeze(x, 1)

        #Flatten input tensor
        x = x.view(x.shape[0],-1)

        # #Compute the convulutional layer output
        # conv_output = self.conv1(x)

        # #Reshape the input for a fully connected layer
        # fully_connected_input = conv_output.view((conv_output.shape[0],-1))
        

        #Compute hidden for fully connected layers
        hidden_layer_output = self.activation(self.FC_input(x))

        #Compute the mean of epoch/sample factor matrix (latent space)
        mean = self.FC_mean(hidden_layer_output)

        #Compute the log_ variance of epoch/sample factor matrix (latent space)
        log_var = self.FC_log_var(hidden_layer_output)

        return mean, log_var 
