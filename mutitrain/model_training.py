import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPModel, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()
        self.loss_history = []

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, y_pred, y_true):
        loss = self.loss_fn(y_pred, y_true)
        self.loss_history.append(loss.item())
        return loss


def batch_gradient_descent_federated(model, optimizer, X, y, learning_rate=0.01, epochs=1, batch_size=32):
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = model.compute_loss(y_pred.flatten(), batch_y)
            loss.backward()
            optimizer.step()

        # 打印每个 epoch 的平均损失
        avg_loss = sum(model.loss_history) / len(model.loss_history) if model.loss_history else float('inf')
        print(f'Client Epoch {epoch}, Loss: {avg_loss}')
        model.loss_history.clear()

    return model, optimizer


def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(X).flatten().numpy()
        mse = ((y.numpy() - y_pred) ** 2).mean()
        print(f'Mean Squared Error: {mse}')



