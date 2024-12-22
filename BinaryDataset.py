import torch
import torch.nn as nn
import pickle

class BinaryDataset(nn.Module):
    def __init__(self, filename):
        super().__init__()
        with open(filename, 'rb') as file:
            self.data = pickle.load(file)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



if __name__ == "__main__":
    dataset = BinaryDataset("./dataset/data.pkl")
    print(dataset[0])

