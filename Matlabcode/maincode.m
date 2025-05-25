clc;
clear;

% 设置参数
N = 10; % 节点数
pe = 0.6; % 边生成概率
min_bandwidth = 10; % 最小带宽
max_bandwidth = 100; % 最大带宽

% 生成随机拓扑图及其带宽，并获取邻接矩阵和代价矩阵

[G, bandwidths, adjacencyMatrix, costMatrix] = generate_random_topology_and_bandwidth(N, pe, min_bandwidth, max_bandwidth);

% 显示结果
disp('邻接矩阵:');
disp(adjacencyMatrix);
disp('代价矩阵:');
disp(costMatrix);
routing(adjacencyMatrix,costMatrix);
figure;
plot(G, 'Layout', 'force', 'LineWidth', 2);
title('随机拓扑图');