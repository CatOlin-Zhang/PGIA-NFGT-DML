function W = DeltaSGD(W, X, D, learning_rate)
    % 假设 X 是单个样本的特征向量，D 是对应的标签
    % 这里实现的是基于单个样本的随机梯度下降更新
    x = X(:); % 将特征向量转换为列向量
    d = D;    % 标签
    y = Sigmoid(W * x); % 预测值
    error = d - y;      % 误差
    gradient = error * x; % 梯度（对单个样本）
    W = W + learning_rate * gradient'; % 更新权重
end
