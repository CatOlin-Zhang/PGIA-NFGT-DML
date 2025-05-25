import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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

    # Convert data to numpy arrays for PCA computation
    X_np = X.numpy()
    y_np = y.numpy().reshape(-1, 1)

    # Compute covariance matrix and perform PCA
    cov_matrix = np.cov(X_np.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Cumulative energy percentage threshold
    energy_threshold = 0.8
    cumulative_energy = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    num_components = np.argmax(cumulative_energy >= energy_threshold) + 1

    # Select principal components
    principal_components = sorted_eigenvectors[:, :num_components]

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0
    batches = 0
    dropped_samples_count = 0  # 记录当前客户端丢弃的样本数量

    projection_history = []  # Debugging purposes

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()

        # Convert batch data to numpy for PCA projection
        batch_X_np = batch_X.numpy()

        # Project each sample onto the principal components
        projections = np.dot(batch_X_np, principal_components)
        projection_history.extend(projections.tolist())  # For debugging

        # Check if any projection value exceeds the threshold
        projection_threshold = 3.5
        valid_samples = np.all(np.abs(projections) <= projection_threshold, axis=1)

        # Calculate dropped samples
        dropped_samples = ~valid_samples
        dropped_samples_count += np.sum(dropped_samples)

        if not np.any(valid_samples):
            continue

        # Filter valid samples
        valid_batch_X = batch_X[valid_samples]
        valid_batch_y = batch_y[valid_samples]

        outputs = model(valid_batch_X)
        loss = criterion(outputs, valid_batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

        # Print average loss for each client epoch
        avg_loss = loss.item() / valid_batch_X.size(0)
        print(f'Client Epoch {epoch}, Loss: {avg_loss}')
        model.loss_history.append(loss.item())

    if batches > 0:
        avg_epoch_loss = total_loss / batches
    else:
        avg_epoch_loss = float('inf')

    return model, avg_epoch_loss, dropped_samples_count


def evaluate_model(model, X, y):
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y.unsqueeze(1))
        print(f'Validation Loss: {loss.item()}')