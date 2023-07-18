from torch.utils.data import Dataset, DataLoader, random_split
import torch

class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.x_data = torch.FloatTensor(data)
        self.y_data = torch.LongTensor(target)
        #self.y_data = self.y_data-1

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]