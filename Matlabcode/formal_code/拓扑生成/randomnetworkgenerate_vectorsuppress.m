%生成随机无向图
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
    
    % 创建空图并添加节点
    G = graph();
    
    % 添加边
    edges = [];
    for i = 1:N-1
        for j = i+1:N
            if rand() < pe
                edges = [edges; i, j];
            end
        end
    end
    
    % 如果没有边，则强制添加至少一条边以避免孤立节点
    if isempty(edges)
        edges = [1, 2];
    end
    
    G = addedge(G, edges(:, 1), edges(:, 2));
    
    % 检查是否有孤立节点
    while any(degree(G) == 0)
        isolatedNodes = find(degree(G) == 0);
        nodeToConnect = isolatedNodes(randi(numel(isolatedNodes)));
        otherNode = randi(N);
        while otherNode == nodeToConnect || ismember(otherNode, neighbors(G, nodeToConnect))
            otherNode = randi(N);
        end
        G = addedge(G, nodeToConnect, otherNode);
    end
    
    % 生成随机带宽
    numEdges = numedges(G);
    totalBandwidth = randi([min_bandwidth, max_bandwidth], 1, numEdges)';
    
    bandwidths = table(G.Edges.EndNodes(:,1), G.Edges.EndNodes(:,2), totalBandwidth, ...
        'VariableNames', {'StartNode', 'EndNode', 'Bandwidth'});
    
    % 获取邻接矩阵
    adjacencyMatrix = zeros(N);
    for k = 1:numEdges
        startNode = G.Edges.EndNodes(k, 1);
        endNode = G.Edges.EndNodes(k, 2);
        bandwidth = totalBandwidth(k);
        adjacencyMatrix(startNode, endNode) = bandwidth;
        adjacencyMatrix(endNode, startNode) = bandwidth; % 无向图对称
    end
    
    % 计算代价矩阵
    costMatrix = zeros(N); % 初始化代价矩阵为0视为不可达
    for k = 1:numEdges
        startNode = G.Edges.EndNodes(k, 1);
        endNode = G.Edges.EndNodes(k, 2);
        bandwidth = totalBandwidth(k);
        if bandwidth > 0
            cost = 1 / (max_bandwidth-bandwidth);
        else
            cost = 0;
        end
        costMatrix(startNode, endNode) = cost;
        costMatrix(endNode, startNode) = cost; % 无向图对称
    end
    
    % 使用Kruskal算法生成最小生成树
    T = minspantree(G);
    
    % 检查生成树的边数是否为N-1
    if numedges(T) ~= N - 1
        disp('此次生成可能包含伪孤立链路');
    else
        disp('随机拓扑生成成功');
    end
end



