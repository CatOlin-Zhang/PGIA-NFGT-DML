function [loss_history_filtered]=PGIA(energy_threshold_of_PGIA,projection_threshold_of_PGIA)


energy_threshold = energy_threshold_of_PGIA;
projection_threshold = projection_threshold_of_PGIA;



% 生成数据
rng(0); % 设置随机种子以确保结果可重复
X1 = 1 * rand(10000, 1);
X2 = 3 * rand(10000, 1);
y = 4 + 1*X1 + 6*X2 + randn(10000, 1);

% 添加x0 = 1到每个实例（用于截距项）
X_b = [ones(10000, 1), X1, X2];

% 超参数设置
eta = 0.0001; % 学习率
max_iterations = 10000; % 最大迭代次数
loss_threshold = 1e-4; % 损失函数的阈值
m = size(X_b, 1); % 样本数量

% 初始化参数
initial_theta = rand(3, 1);
theta = initial_theta;

% 实验1计时器
time_of_exp1=tic;

% 计算协方差矩阵及其特征值和特征向量
cov_matrix = cov(X_b(:, 2:end)); % 只考虑特征部分
[eigenvectors, eigenvalues] = eig(cov_matrix);
sorted_eigenvalues = diag(eigenvalues);
[~, idx] = sort(sorted_eigenvalues, 'descend');
sorted_eigenvalues = sorted_eigenvalues(idx);
eigenvectors = eigenvectors(:, idx);

% 累积能量百分比阈值
%  energy_threshold = 0.9;
cumulative_energy = cumsum(sorted_eigenvalues) / sum(sorted_eigenvalues);
num_components = find(cumulative_energy >= energy_threshold, 1, 'first');

% 主成分选择
principal_components = eigenvectors(:, 1:num_components);

% 第五次随机梯度下降使用PCAcov方案
iteration_count = 0;
projection_history= [];
loss_history_filtered = [];
previous_loss = Inf;

while iteration_count < max_iterations
    random_index = randi(m);
    xi = X_b(random_index, :);
    yi = y(random_index);
    x_sample = xi(2:end); % 提取特征部分
    
    % 投影到主成分上
    projection = principal_components' .* x_sample;
    projection_history=[projection_history,projection];
    
    % 判断是否用于模型更新
    if any(abs(projection) > projection_threshold) 
        gradient = 2 * xi' * (xi * theta - yi);
        theta = theta - eta * gradient;
        iteration_count = iteration_count + 1;
        
        % 计算当前损失函数值
        predictions = X_b * theta;
        current_loss = mean((predictions - y).^2);
        loss_history_filtered = [loss_history_filtered, current_loss];
        
        % 检查损失函数是否收敛
        if current_loss < loss_threshold
            fprintf('第五次在新最大迭代次数内收敛于第 %d 次迭代，损失函数值为 %.6f\n', iteration_count, current_loss);
            fprintf('拟合的超平面方程为: y = %.2f + %.2fx1 + %.2fx2\n', theta(1), theta(2), theta(3));
            break;
        end
        
        previous_loss = current_loss;
    end
end

% 如果第五次循环结束后仍未收敛，输出最后一次的损失函数值和拟合方程
if iteration_count == max_iterations
    final_loss = loss_history_filtered(end);
    fprintf('第五次在达到最大迭代次数 %d 后，损失函数值为 %.6f\n', max_iterations, final_loss);
    fprintf('拟合的超平面方程为: y = %.2f + %.2fx1 + %.2fx2\n', theta(1), theta(2), theta(3));
end

toc(time_of_exp1);
end
% 
% % 绘制结果
% figure;
% scatter3(X1, X2, y, 'filled', 'DisplayName', 'Data points'); % 原始数据点
% hold on;
% 
% % 创建网格以便绘制拟合超平面
% [X1_grid, X2_grid] = meshgrid(min(X1):0.1:max(X1), min(X2):0.1:max(X2));
% X_b_grid = [ones(numel(X1_grid), 1), X1_grid(:), X2_grid(:)];
% y_predict_grid = X_b_grid * theta;
% y_predict_grid = reshape(y_predict_grid, size(X1_grid));
% 
% surf(X1_grid, X2_grid, y_predict_grid, 'FaceAlpha', 0.5, 'DisplayName', 'Predictions'); % 拟合超平面
% xlabel('X1');
% ylabel('X2');
% zlabel('y');
% title('SGD Linear Regression in 3D with PCA');
% legend show;
% grid on;
% view(3); % 设置视角为3D
% 
% % 绘制第五次实验的损失函数历史
% figure;
% plot(loss_history_filtered_5th, 'b', 'DisplayName', 'PGIA');
% xlabel('Iteration');
% ylabel('Loss');
% title('第五次实验的SGD损失函数历史');
% legend show;
% grid on;
% 