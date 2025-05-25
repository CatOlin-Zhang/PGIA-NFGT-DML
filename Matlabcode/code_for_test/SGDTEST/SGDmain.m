clc;
clear;
X = [0 0 1; 0 1 1; 1 0 1; 1 1 1];
D = [0; 0; 1; 1];
num_epochs = 1000; % 设定总的 epoch 数
learning_rate = 0.01; % 设置学习率
W1 = 2*rand(1, 3) - 1; % 初始化权重
E1 = zeros(num_epochs, 1); % 初始化误差数组
E2 = zeros(num_epochs-1,1);%损失比例数组

for epoch = 1:num_epochs
    % 随机打乱数据集（可选，但有助于收敛）
    % [X, idx] = datasample(X, size(X, 1), 'Replace', false);
    % D = D(idx);
    % 遍历整个数据集（这里实际上是小批量梯度下降的一个特例，即批量大小为1）
    for i = 1:size(X, 1)
        W1 = DeltaSGD(W1, X(i, :), D(i), learning_rate); % 使用当前样本更新权重
    end
    total_error = 0;
    for i = 1:size(X, 1)
        x = X(i, :)';
        d = D(i);
        v1 = W1 * x;
        y1 = Sigmoid(v1);
        total_error = total_error + (d - y1)^2;
    end
    E1(epoch) = total_error / size(X, 1); % 存储平均误差
    
    % 可选：显示当前 epoch 的平均误差
    if mod(epoch, 100) == 0
        fprintf('Epoch %d: Average Error = %.4f\n', epoch, E1(epoch));
       E2(epoch)= E1(epoch-1)/E1(epoch);
    end
end

% 绘制误差曲线
plot(E1, 'r')
xlabel('Epoch')
ylabel('Average Training Error')
title('Average Training Error Over Epochs using SGD')
legend('SGD')
hold on
plot(E2)
hold off