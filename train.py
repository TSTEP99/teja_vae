"""File used to train VAE-CP"""
from data import TensorDataset
from losses import original_loss, total_variation_loss
from math import floor
from preprocess import create_indices, process_eegs
from torchmetrics import MeanSquaredError
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from teja_vae import teja_vae
import torch

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    model.train()
    for batch, samples in tqdm(enumerate(dataloader)):
        # Compute prediction and loss
        tensor_mean, tensor_log_var = model(samples)
        loss = loss_fn(samples, tensor_mean, tensor_log_var, model.mus, model.lambdas, model.mus_tildes, model.lambdas_tildes)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    HIDDEN_LAYER_SIZE = 100
    RANK = 7

    full_psds, _, _, _, _, grade, epi_dx, alz_dx, _, _, _, _ = process_eegs()

    pop_psds= full_psds[(epi_dx<0) & (alz_dx<0)]

    pop_psds /= (torch.max(pop_psds) - torch.min(pop_psds))

    indices = create_indices(pop_psds.shape)
    indices = indices.to(torch.long)
    indices = indices.to(DEVICE)

    dims = pop_psds.shape

    print("Dimensions of population tensor:", dims)

    pop_psds = pop_psds.to(DEVICE)

    total_length = pop_psds.shape[0]

    train_length = floor(0.8 * total_length)
    val_length = floor( 0.5 * (total_length-train_length))
    test_length = total_length - train_length - val_length


    lengths = [train_length, val_length, test_length]

    dataset= TensorDataset(pop_psds, indices)

    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths, generator = torch.Generator().manual_seed(42))

    print(f"Training Set has length {train_dataset.__len__()}")
    print(f"Validation Set has length {val_dataset.__len__()}")
    print(f"Test Set has length {test_dataset.__len__()}")

    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

    other_dims = dims[1:] 

    model = teja_vae(other_dims, encoder_hidden_layer_size = HIDDEN_LAYER_SIZE, decoder_hidden_layer_size = HIDDEN_LAYER_SIZE, rank = RANK, device = DEVICE)

    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, betas = (0.9, 0.999), eps=1e-8)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, total_variation_loss, optimizer)
        test_loop(test_dataloader, model, total_variation_loss)
        torch.save(model, f'checkpoints/vae_cp_epoch_{t+1}.pth')
    print("Done!")