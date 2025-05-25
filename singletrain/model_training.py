import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        # Xavier initialization
        std_dev = np.sqrt(2. / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size) * std_dev
        self.bias = np.zeros((1, output_size))
        self.dweights = None
        self.dbias = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, dout):
        self.dweights = np.dot(self.X.T, dout)
        self.dbias = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.weights.T)

class MLPModel:
    def __init__(self, input_size, hidden_sizes, output_size):
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(DenseLayer(prev_size, size))
            prev_size = size
        layers.append(DenseLayer(prev_size, output_size))
        self.layers = layers

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A.flatten()

    def backward(self, dout, X):
        dA = dout.reshape(-1, 1)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dweights
            layer.bias -= learning_rate * layer.dbias

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

def batch_gradient_descent(model, X_train, y_train, learning_rate=0.001, epochs=1000, batch_size=32):
    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            # Forward pass
            y_pred = model.forward(X_batch)

            # Compute loss and derivative
            loss = mean_squared_error(y_batch, y_pred)
            dy_pred = mean_squared_error_derivative(y_batch, y_pred)

            # Backward pass
            model.backward(dy_pred, X_batch)

            # Update parameters
            model.update_parameters(learning_rate)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.forward(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f'Mean Squared Error: {mse}')
    return mse



