import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from tqdm import tqdm
from Loss import *
from Visualizer import Visualizer


class ModelTrainer():
    def __init__(self, model, optimizer='adam', lr=0.001):
        super().__init__()
        self.model = model
        self.vis = Visualizer()
        if optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(), lr=lr)

        self.train_acc = []
        self.val_acc = []

    def train(self, criterion, train_loader, num_epochs):
        for epoch in range(0, num_epochs):
            print(f"epoch #{epoch}")
            self.train_one_epoch(criterion=criterion, train_loader=train_loader, epoch_id=epoch, num_epochs=num_epochs)

    def train_and_validation(self, criterion, train_loader, val_loader, num_epochs):
        for epoch in range(0, num_epochs):
            print(f"epoch #{epoch}")
            self.train_one_epoch(criterion=criterion, train_loader=train_loader, epoch_id=epoch, num_epochs=num_epochs)
            self.validate(criterion=criterion, val_loader=val_loader, epoch_id=epoch, num_epochs=num_epochs)

    def train_one_epoch(self, criterion, train_loader, epoch_id, num_epochs):
        acc_num = 0
        totel_num = 0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch_id + 1}/{num_epochs}")
        self.model.train()
        for data in pbar:
            feature = self.model(data)
            self.optimizer.zero_grad()
            loss = criterion(feature, data["result"])
            loss.backward()
            self.optimizer.step()

            prob = torch.softmax(feature, dim=-1)
            pred = torch.argmax(prob, dim=-1)
            acc_num += torch.sum(pred == data["result"]).item()
            totel_num += data["result"].shape[0]

            pbar.set_postfix(loss = loss.item())

        self.train_acc.append(acc_num/totel_num)

    def save_model(self, epoch_ID):
        torch.save(self.model.state_dict(), "./model/epoch_" + str(epoch_ID) + "_model.pt")
        print(f'Model saved to {"./model/epoch_" + str(epoch_ID) + "_model.pt"}')

    def validate(self, criterion, val_loader, epoch_id, num_epochs, model_path=None):
        if not val_loader:
            print("No validation")
            return
        if model_path:
            self.model.load(model_path)

        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch_id + 1}/{num_epochs}")
        self.model.eval()  # 设置模型为评估模式
        acc_num = 0
        totel_num = 0
        with torch.no_grad():
            for data in pbar:
                feature = self.model(data)
                loss = criterion(feature, data["result"])

                prob = torch.softmax(feature, dim=-1)
                pred = torch.argmax(prob, dim=-1)
                acc_num += torch.sum(pred == data["result"]).item()
                totel_num += data["result"].shape[0]
                pbar.set_postfix(loss = loss.item())

        self.val_acc.append(acc_num / totel_num)

    def get_acc(self):
        return {"train": self.train_acc, "val": self.val_acc}

if __name__ == "__main__":
    a = []
    x = torch.tensor([1,2,3])
    a += x.detach().numpy().tolist()
    print(a)
