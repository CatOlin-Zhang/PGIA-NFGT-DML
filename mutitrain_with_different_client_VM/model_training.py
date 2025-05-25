import torch
import torch.nn as nn
import torch.optim as optim


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPModel, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
        self.loss_history = []

    def forward(self, x):
        return self.model(x)


def batch_gradient_descent_federated(model, optimizer, X, y, learning_rate, batch_size, epoch):
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0
    batches = 0

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()

        # Calculate gradient norm
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        # Gradient threshold
        gradient_threshold = -1  # You can adjust this threshold

        if grad_norm > gradient_threshold:
            optimizer.step()

            total_loss += loss.item()
            batches += 1

            # Print average loss for each client epoch
            avg_loss = loss.item() / batch_X.size(0)
            print(f'Client Epoch {epoch}, Loss: {avg_loss}')
            model.loss_history.append(loss.item())

    if batches > 0:
        avg_epoch_loss = total_loss / batches
    else:
        avg_epoch_loss = float('inf')

    return model, avg_epoch_loss


def evaluate_model(model, X, y):
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y.unsqueeze(1))
        print(f'Validation Loss: {loss.item()}')

# 其他函数保持不变



