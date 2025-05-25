import numpy as np
import torch

from data_preprocessing import load_data, preprocess_data, log_transform, inverse_log_transform
from model_training import MLPModel, batch_gradient_descent_federated, evaluate_model
from prediction import predict_and_submit
import pickle
from collections import OrderedDict


def main():
    # 加载数据
    train_path = 'train.csv'
    test_path = 'test.csv'
    train_data, test_data = load_data(train_path, test_path)

    # 为训练数据和测试数据分别添加新特征
    train_data['TotalSF'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']
    test_data['TotalSF'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']

    # 数据预处理
    preprocessor, numeric_features, categorical_features = preprocess_data(train_data)
    X_train_processed = preprocessor.fit_transform(train_data)
    X_test_processed = preprocessor.transform(test_data)

    # 将稀疏矩阵转换为密集矩阵
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # 对目标变量进行对数变换
    y_train_log = log_transform(train_data['SalePrice'].values)

    # 创建虚拟客户端
    num_clients = 4
    client_ids = [f"client_{i}" for i in range(num_clients)]
    clients = {client_id: [] for client_id in client_ids}

    # 划分数据到不同客户端
    for i in range(len(X_train_processed)):
        clients[client_ids[i % num_clients]].append(i)

    datasets = {}
    for client_id, indices in clients.items():
        X_client = X_train_processed[indices]
        y_client = y_train_log[indices]
        datasets[client_id] = (X_client, y_client)

    # 创建MLP模型
    input_size = X_train_processed.shape[1]
    hidden_sizes = [64, 32]  # 两层隐藏层，每层分别有64和32个神经元
    output_size = 1
    global_model = MLPModel(input_size, hidden_sizes, output_size)

    # 训练模型
    epochs = 1000
    learning_rate = 0.001
    batch_size = 32

    for epoch in range(epochs):
        new_models = []
        for client_id in client_ids:
            X_client, y_client = datasets[client_id]
            local_model = MLPModel(input_size, hidden_sizes, output_size)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)

            # Convert data to tensors
            X_client_tensor = torch.tensor(X_client, dtype=torch.float32)
            y_client_tensor = torch.tensor(y_client, dtype=torch.float32)

            local_model, _ = batch_gradient_descent_federated(
                local_model, optimizer, X_client_tensor, y_client_tensor, learning_rate=learning_rate,
                batch_size=batch_size
            )
            new_models.append(local_model)

        # 模型聚合
        with torch.no_grad():
            averaged_params = OrderedDict()
            for name, param in global_model.named_parameters():
                params = torch.stack([model.state_dict()[name] for model in new_models])
                averaged_params[name] = torch.mean(params, dim=0)
            global_model.load_state_dict(averaged_params)

        # 打印损失
        total_loss = 0
        total_batches = 0
        for model in new_models:
            if model.loss_history:
                total_loss += sum(model.loss_history)
                total_batches += len(model.loss_history)
                model.loss_history.clear()

        if total_batches > 0:
            avg_loss = total_loss / total_batches
            print(f'Epoch {epoch}, Loss: {avg_loss}')
        else:
            print(f'Epoch {epoch}, No loss recorded')

    # 评估模型
    val_size = int(0.2 * len(y_train_log))
    X_val_processed = X_train_processed[-val_size:]
    y_val_log = y_train_log[-val_size:]

    # Convert validation data to tensors
    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_log, dtype=torch.float32)

    evaluate_model(global_model, X_val_tensor, y_val_tensor)

    # 预测并生成提交文件
    submission_path = 'submission.csv'
    predict_and_submit(global_model, test_data, submission_path, preprocessor)

    # 保存模型
    with open('mlp_model.pkl', 'wb') as f:
        pickle.dump(global_model, f)


if __name__ == "__main__":
    main()



