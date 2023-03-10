from torch.utils.data import Dataset

class TensorDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    
    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        return self.tensor[idx]