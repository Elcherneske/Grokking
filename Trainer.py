import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
import time
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
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
            self.train_one_epoch(criterion=criterion, train_loader=train_loader)

    def train_and_validation(self, criterion, train_loader, val_loader, num_epochs):
        for epoch in range(0, num_epochs):
            print(f"epoch #{epoch}")
            self.train_one_epoch(criterion=criterion, train_loader=train_loader)
            self.validate(criterion=criterion, val_loader=val_loader)

    def train_one_epoch(self, criterion, train_loader):
        acc_num = 0
        totel_num = 0
        self.model.train()
        for n_step, data in enumerate(train_loader):
            feature = self.model(data)
            self.optimizer.zero_grad()
            loss = criterion(feature, data["label"])
            loss.backward()
            self.optimizer.step()

            with torch.no_grad:
                prob = torch.softmax(feature, dim=-1)
                pred = torch.argmax(prob, dim=-1)
                acc_num += torch.sum(pred == data["label"]).item()
                totel_num += data["label"].size(0)

            if n_step % 100 == 0:
                print(f"n_step: {n_step}, Loss: {loss.item()}")

        self.train_acc.append(acc_num/totel_num)

    def save_model(self, epoch_ID):
        torch.save(self.model.state_dict(), "./model/epoch_" + str(epoch_ID) + "_model.pt")
        print(f'Model saved to {"./model/epoch_" + str(epoch_ID) + "_model.pt"}')

    def validate(self, criterion, val_loader, model_path=None):
        if not val_loader:
            print("No validation")
            return

        if model_path:
            self.model.load(model_path)
        self.model.eval()  # 设置模型为评估模式
        print("Start validation")

        acc_num = 0
        totel_num = 0
        with torch.no_grad():
            for n_step, data in enumerate(val_loader):
                feature = self.model(data)
                loss = criterion(feature, data["label"])

                prob = torch.softmax(feature, dim=-1)
                pred = torch.argmax(prob, dim=-1)
                acc_num += torch.sum(pred == data["label"]).item()
                totel_num += data["label"].size(0)

                if n_step % 100 == 0:
                    print(f"n_step: {n_step}, Loss: {loss.item()}")
            self.val_acc.append(acc_num / totel_num)


if __name__ == "__main__":
    a = []
    x = torch.tensor([1,2,3])
    a += x.detach().numpy().tolist()
    print(a)
