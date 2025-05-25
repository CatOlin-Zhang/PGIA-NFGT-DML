% 生成样本数据
rng(1); % 设置随机种子以确保结果可重复
n = 100; % 样本数量
X1 = rand(n, 1) * 10; % 特征1范围为0到10
X2 = rand(n, 1) * 10; % 特征2范围为0到10

% 添加一个常数项（截距）
X = [ones(n, 1), X1, X2];

% 真实系数
beta_true = [5; -2; 3]; % 包括截距

% 模拟响应变量Y
epsilon = 0.5 * randn(n, 1); % 随机误差
Y = X * beta_true + epsilon;

% 多元线性回归分析
mdl = fitlm(X(:, [1, 2]), Y);

% 显示回归模型摘要信息
disp(mdl)

% 创建网格用于绘制平面
[X1_grid, X2_grid] = meshgrid(linspace(min(X1), max(X1), 50), linspace(min(X2), max(X2), 50));
X_grid = [ones(size(X1_grid(:))), X1_grid(:), X2_grid(:)];

% 计算实际的Z值
Z_actual = X_grid * beta_true;

% 计算预测的Z值
Z_pred = predict(mdl, X_grid(:, [1, 2]));

% 调整形状以便绘图
Z_actual = reshape(Z_actual, size(X1_grid));
Z_pred = reshape(Z_pred, size(X1_grid));

% 绘制三维图
figure;
scatter3(X1, X2, Y, 'filled', 'DisplayName', 'Data Points');
hold on;

% 绘制实际的平面
surf(X1_grid, X2_grid, Z_actual, 'FaceAlpha', 0.5, 'DisplayName', 'Actual Plane');

% 绘制预测的平面
surf(X1_grid, X2_grid, Z_pred, 'FaceAlpha', 0.5, 'DisplayName', 'Predicted Plane');

xlabel('X1');
ylabel('X2');
zlabel('Y');
title('Actual vs Predicted Planes');
legend show;
grid on;
view(3); % 设置视角为三维视角
colorbar; % 添加颜色条
hold off;



