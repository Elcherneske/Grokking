import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle

class BinaryDataset(nn.Module):
    def __init__(self, filename, is_train=True, train_ratio = 0.8, device='cpu'):
        super().__init__()
        self.device = device
        with open(filename, 'rb') as file:
            self.data = pickle.load(file)

        if is_train:
            number = round(len(self.data) * train_ratio)
            self.data = self.data[:number]
        else:
            number = round(len(self.data) * train_ratio)
            self.data = self.data[number:]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.data[index]['x'] = torch.tensor(self.data[index]['x']).to(self.device)
        self.data[index]['op'] = torch.tensor(self.data[index]['op']).to(self.device)
        self.data[index]['y'] = torch.tensor(self.data[index]['y']).to(self.device)
        self.data[index]['='] = torch.tensor(self.data[index]['=']).to(self.device)
        self.data[index]['result'] = torch.tensor(self.data[index]['result']).to(self.device)
        return self.data[index]



if __name__ == "__main__":
    dataset = BinaryDataset("./dataset/data.pkl")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    for index, data in enumerate(dataloader):
        print(data)
        if index > 1:
            break

