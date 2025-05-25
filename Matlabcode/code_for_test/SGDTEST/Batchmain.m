clear;
X = [0 0 1; 0 1 1; 1 0 1; 1 1 1]; % 特征矩阵
D = [0; 0; 1; 1];                  % 标签向量
num_epochs = 1000;                 % 迭代次数
learning_rate = 0.1;               % 学习率
W = 2*rand(1, 3) - 1;              % 初始化权重
E = zeros(num_epochs, 1);          % 初始化误差数组
 
% 批量梯度下降训练
for epoch = 1:num_epochs
    W = DeltaBatch(W, X, D, learning_rate); % 使用整个数据集更新权重
    
    % 计算并存储整个数据集的平均误差（可选，用于监控训练过程）
    total_error = 0;
    for i = 1:size(X, 1)
        x = X(i, :)';
        d = D(i);
        v = W * x;
        y = Sigmoid(v);
        total_error = total_error + (d - y)^2;
    end
    disp(size(X, 1))
    E(epoch) = total_error / 4; % 存储平均误差
    
    % 显示当前 epoch 的平均误差（可选）
    if mod(epoch, 100) == 0
        fprintf('Epoch %d: Average Error = %.4f\n', epoch, E(epoch));
    end
end
 
% 绘制误差曲线
plot(E, 'b')
xlabel('Epoch')
ylabel('Average Training Error')
title('Average Training Error Over Epochs using Batch Gradient Descent')
legend('Batch GD')
hold off