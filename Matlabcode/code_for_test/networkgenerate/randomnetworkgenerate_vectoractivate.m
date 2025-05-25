clc;
clear;

% 设置参数
N = 10; % 节点数
pe = 0.3; % 边生成概率
min_bandwidth = 10; % 最小带宽
max_bandwidth = 100; % 最大带宽

% 生成随机拓扑图及其带宽，并获取邻接矩阵和代价矩阵
[G, bandwidths, adjacencyMatrix, costMatrix] = generate_random_topology_and_bandwidth(N, pe, min_bandwidth, max_bandwidth);

% 显示结果
disp('邻接矩阵:');
disp(adjacencyMatrix);
disp('代价矩阵:');
disp(costMatrix);

figure;
plot(G, 'Layout', 'force', 'LineWidth', 2);
title('随机拓扑图');

function [G, bandwidths, adjacencyMatrix, costMatrix] = generate_random_topology_and_bandwidth(N, pe, min_bandwidth, max_bandwidth)
    % 生成一个随机拓扑图，并为其每条边分配随机带宽。
    % 参数:
    % N -- 节点数量
    % pe -- 边存在的概率
    % min_bandwidth -- 带宽范围的最小值
    % max_bandwidth -- 带宽范围的最大值
    
    if N <= 0
        error('节点数量必须大于0');
    end
    
    if min_bandwidth >= max_bandwidth
        error('最小带宽必须小于最大带宽');
    end
    
    % 创建空有向图
    G = digraph();
    
    % 添加边
    edges = [];
    for i = 1:N
        for j = 1:N
            if i ~= j && rand() < pe
                edges = [edges; i, j];
            end
        end
    end
    
    % 如果没有边，则强制添加至少一条边以避免孤立节点
    if isempty(edges)
        edges = [1, 2];
    end
    
    G = addedge(G, edges(:, 1), edges(:, 2));
    
   
    % 生成随机带宽
    numEdges = numedges(G);
    totalBandwidthForward = randi([min_bandwidth, max_bandwidth], 1, numEdges)';
    totalBandwidthBackward = randi([min_bandwidth, max_bandwidth], 1, numEdges)';
    
    bandwidths = table(G.Edges.EndNodes(:,1), G.Edges.EndNodes(:,2), totalBandwidthForward, totalBandwidthBackward, ...
        'VariableNames', {'StartNode', 'EndNode', 'ForwardBandwidth', 'BackwardBandwidth'});
    
    % 获取邻接矩阵
    adjacencyMatrix = zeros(N);
    for k = 1:numEdges
        startNode = G.Edges.EndNodes(k, 1);
        endNode = G.Edges.EndNodes(k, 2);
        forwardBandwidth = totalBandwidthForward(k);
        backwardBandwidth = totalBandwidthBackward(k);
        adjacencyMatrix(startNode, endNode) = forwardBandwidth;
        adjacencyMatrix(endNode, startNode) = backwardBandwidth; % 反向带宽
    end
    
    % 计算代价矩阵
    costMatrix = zeros(N); % 初始化代价矩阵为0
    for k = 1:numEdges
        startNode = G.Edges.EndNodes(k, 1);
        endNode = G.Edges.EndNodes(k, 2);
        forwardBandwidth = totalBandwidthForward(k);
        backwardBandwidth = totalBandwidthBackward(k);
        if forwardBandwidth > 0
            costMatrix(startNode, endNode) = 1 / (max_bandwidth - forwardBandwidth);
        else
            costMatrix(startNode, endNode) = 0; % 设置为0
        end
        if backwardBandwidth > 0
            costMatrix(endNode, startNode) = 1 / (max_bandwidth - backwardBandwidth);
        else
            costMatrix(endNode, startNode) = 0; % 设置为0
        end
    end
    
    % 使用Dijkstra算法生成最短路径树
    T = shortestpathtree(G, 1); % 从节点1开始的最短路径树
    
    % 检查生成树的边数是否为N-1
    if numedges(T) ~= N - 1
        disp('此次生成可能包含伪孤立链路');
    else
        disp('随机有向拓扑生成成功');
    end
end



