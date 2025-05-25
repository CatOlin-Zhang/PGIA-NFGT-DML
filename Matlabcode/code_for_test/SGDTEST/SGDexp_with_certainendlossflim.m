clc;
clear;

% 生成数据
rng(0); % 设置随机种子以确保结果可重复
X1 = 2 * rand(10000, 1);
X2 = 3 * rand(10000, 1);
y = 4 + 2*X1 + 3*X2 + randn(10000, 1);

% 截距
X_b = [ones(10000, 1), X1, X2];

% 超参数设置
eta = 0.0001; % 学习率
max_iterations = 20000; % 最大迭代次数
loss_threshold = 1e-5; % 损失函数的阈值
m = size(X_b, 1); % 样本数量

% 初始化参数
initial_theta = rand(3, 1);
theta = initial_theta;

% 初始化损失函数值
loss_history = [];

% 第一次随机梯度下降以找到收敛次数
for iteration = 1:max_iterations
    random_index = randi(m);
    xi = X_b(random_index, :);
    yi = y(random_index);
    gradient = 2 * xi' * (xi * theta - yi);
    theta = theta - eta * gradient;
    
    % 计算当前损失函数值
    predictions = X_b * theta;
    current_loss = mean((predictions - y).^2);
    loss_history = [loss_history, current_loss];
    
    % 检查损失函数是否小于阈值
    if current_loss < loss_threshold
        convergence_iteration = iteration;
        fprintf('第一次收敛于第 %d 次迭代，损失函数值为 %.6f\n', convergence_iteration, current_loss);
        fprintf('拟合的超平面方程为: y = %.2f + %.2fx1 + %.2fx2\n', theta(1), theta(2), theta(3));
        break;
    end
end

% 如果没有在最大迭代次数内收敛，设置收敛次数为最大迭代次数
if ~exist('convergence_iteration', 'var')
    convergence_iteration = max_iterations;
    fprintf('未在最大迭代次数内收敛，使用最大迭代次数 %d\n', convergence_iteration);
end

% 使用收敛次数的0.8倍作为新的最大迭代次数
new_max_iterations = floor(convergence_iteration * 0.8);
fprintf('使用新的最大迭代次数: %d\n', new_max_iterations);

% 重新初始化参数为第一次SGD之前的初始值
theta = initial_theta;

% 初始化损失函数值
loss_history_new = [];

% 第二次随机梯度下降使用新的最大迭代次数
for iteration = 1:new_max_iterations
    random_index = randi(m);
    xi = X_b(random_index, :);
    yi = y(random_index);
    gradient = 2 * xi' * (xi * theta - yi);
    theta = theta - eta * gradient;
    
    % 计算当前损失函数值
    predictions = X_b * theta;
    current_loss = mean((predictions - y).^2);
    loss_history_new = [loss_history_new, current_loss];
    
    % 检查损失函数是否小于阈值
    if current_loss < loss_threshold
        fprintf('第二次在新最大迭代次数内收敛于第 %d 次迭代，损失函数值为 %.6f\n', iteration, current_loss);
        fprintf('拟合的超平面方程为: y = %.2f + %.2fx1 + %.2fx2\n', theta(1), theta(2), theta(3));
        break;
    end
end

% 如果第二次循环结束后仍未收敛，输出最后一次的损失函数值和拟合方程
if length(loss_history_new) == new_max_iterations
    final_loss = loss_history_new(end);
    fprintf('第二次在达到最大迭代次数 %d 后，损失函数值为 %.6f\n', new_max_iterations, final_loss);
    fprintf('拟合的超平面方程为: y = %.2f + %.2fx1 + %.2fx2\n', theta(1), theta(2), theta(3));
end

% 重新初始化参数为第一次SGD之前的初始值
theta = initial_theta;

% 初始化损失函数值
loss_history_filtered = [];
grad_norms_third = []; % 存储梯度范数

% 第三次随机梯度下降使用与第二次相同的最大迭代次数
iteration_count = 0;

while iteration_count < new_max_iterations
    random_index = randi(m);
    xi = X_b(random_index, :);
    yi = y(random_index);
    gradient = 2 * xi' * (xi * theta - yi);
    grad_norm = norm(gradient);
    grad_norms_third = [grad_norms_third, grad_norm];
    grad_norms_sorted = sort(grad_norms_third, 'descend');
    p_percentile = ceil(length(grad_norms_sorted) * 0.75); % 前75%
    threshold = grad_norms_sorted(p_percentile);
    
    % 检查梯度是否大于阈值
    if grad_norm >= threshold
        theta = theta - eta * gradient;
        iteration_count = iteration_count + 1;
        
        % 计算当前损失函数值
        predictions = X_b * theta;
        current_loss = mean((predictions - y).^2);
        loss_history_filtered = [loss_history_filtered, current_loss];
        
        % 检查损失函数是否小于阈值
        if current_loss < loss_threshold
            fprintf('第三次在新最大迭代次数内收敛于第 %d 次迭代，损失函数值为 %.6f\n', iteration_count, current_loss);
            fprintf('拟合的超平面方程为: y = %.2f + %.2fx1 + %.2fx2\n', theta(1), theta(2), theta(3));
            break;
        end
    end
end

% 如果第三次循环结束后仍未收敛，输出最后一次的损失函数值和拟合方程
if iteration_count == new_max_iterations
    final_loss = loss_history_filtered(end);
    fprintf('第三次在达到最大迭代次数 %d 后，损失函数值为 %.6f\n', new_max_iterations, final_loss);
    fprintf('拟合的超平面方程为: y = %.2f + %.2fx1 + %.2fx2\n', theta(1), theta(2), theta(3));
end

% 重新初始化参数为第一次SGD之前的初始值
theta = initial_theta;

% 初始化损失函数值
loss_history_filtered_4th = [];
grad_norms_fourth = []; % 存储梯度范数

% 第四次随机梯度下降使用与第二次相同的最大迭代次数
iteration_count = 0;

while iteration_count < new_max_iterations
    random_index = randi(m);
    xi = X_b(random_index, :);
    yi = y(random_index);
    gradient = 2 * xi' * (xi * theta - yi);
    grad_norm = norm(gradient);
    grad_norms_fourth = [grad_norms_fourth, grad_norm];
    grad_norms_sorted = sort(grad_norms_fourth, 'descend');
    p_percentile = ceil(length(grad_norms_sorted) * 0.5); % 前50%
    threshold = grad_norms_sorted(p_percentile);
    
    % 检查梯度是否大于阈值
    if grad_norm >= threshold
        theta = theta - eta * gradient;
        iteration_count = iteration_count + 1;
        
        % 计算当前损失函数值
        predictions = X_b * theta;
        current_loss = mean((predictions - y).^2);
        loss_history_filtered_4th = [loss_history_filtered_4th, current_loss];
        
        % 检查损失函数是否小于阈值
        if current_loss < loss_threshold
            fprintf('第四次在新最大迭代次数内收敛于第 %d 次迭代，损失函数值为 %.6f\n', iteration_count, current_loss);
            fprintf('拟合的超平面方程为: y = %.2f + %.2fx1 + %.2fx2\n', theta(1), theta(2), theta(3));
            break;
        end
    end
end

% 如果第四次循环结束后仍未收敛，输出最后一次的损失函数值和拟合方程
if iteration_count == new_max_iterations
    final_loss = loss_history_filtered_4th(end);
    fprintf('第四次在达到最大迭代次数 %d 后，损失函数值为 %.6f\n', new_max_iterations, final_loss);
    fprintf('拟合的超平面方程为: y = %.2f + %.2fx1 + %.2fx2\n', theta(1), theta(2), theta(3));
end

% 绘制结果
figure;
scatter3(X1, X2, y, 'filled', 'DisplayName', 'Data points'); % 原始数据点
hold on;

% 创建网格绘制拟合超平面
[X1_grid, X2_grid] = meshgrid(min(X1):0.1:max(X1), min(X2):0.1:max(X2));
X_b_grid = [ones(numel(X1_grid), 1), X1_grid(:), X2_grid(:)];
y_predict_grid = X_b_grid * theta;
y_predict_grid = reshape(y_predict_grid, size(X1_grid));

surf(X1_grid, X2_grid, y_predict_grid, 'FaceAlpha', 0.5, 'DisplayName', 'Predictions'); % 拟合超平面
xlabel('X1');
ylabel('X2');
zlabel('y');
title('SGD Linear Regression in 3D');
legend show;
grid on;
view(3); % 设置视角为3D

% 合并绘制第一次、第二次、第三次和第四次损失函数历史
figure;
plot(loss_history, 'b', 'DisplayName', '第一次SGD损失函数结果');
hold on;
plot(loss_history_new, 'r', 'DisplayName', '第二次限定迭代次数SGD结果');
plot(loss_history_filtered, 'g', 'DisplayName', '第三次一定阈值丢失SGD结果');
plot(loss_history_filtered_4th, 'm', 'DisplayName', '第四次更大阈值丢失SGD结果');
xlabel('Iteration');
ylabel('Loss');
title('SGD损失函数历史');
legend show;
grid on;



