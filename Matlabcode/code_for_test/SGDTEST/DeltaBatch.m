function W = DeltaBatch(W, X, D, learning_rate)
    % X 是特征矩阵，每一行是一个样本
    % D 是标签向量
    % learning_rate 是学习率
    % W 是权重向量
    % 初始化梯度为零向量
    gradient = zeros(size(W));
    
    % 遍历整个数据集计算梯度
    for i = 1:size(X, 1)
        x = X(i, :)'; % 取出一个样本的特征向量，并转置为列向量
        d = D(i);     % 取出对应的标签
        y = Sigmoid(W * x); % 计算预测值
        error = d - y;      % 计算误差
        % 累积梯度
        gradient = gradient + error * x;
    end
    
    % 计算平均梯度并更新权重
    average_gradient = gradient / size(X, 1);
    W = W + learning_rate * average_gradient';
end