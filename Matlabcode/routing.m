function [flow, totalCost] = routing(V, C)
    % V: 容量矩阵
    % C: 消耗矩阵
    n = size(V, 1); % 节点数量
    [numNodes, ~] = size(V);
% 初始化边列表和权重列表
s = [];
t = [];
weights = [];
% 遍历邻接矩阵以提取边和权重
for i = 1:numNodes
    for j = 1:numNodes
        if V(i, j) ~= 0 % 如果存在边
            s = [s; i]; % 起点
            t = [t; j]; % 终点
            weights = [weights; V(i, j)]; % 权重
        end
    end
end
% 创建有向图对象
G = digraph(s, t, weights);
% 绘制有向图，并显示边的权重
plot(G, 'EdgeLabel', G.Edges.Weight, 'Layout', 'layered');
% 计算从节点 1 到节点 n 的最大流
mf = maxflow(G, 1, n);
disp('Matlab最大流计算结果：');
disp(mf);
    % 初始化流量矩阵和总成本
    flow = zeros(n);
    totalCost = 0;
    
    % 源节点和汇节点
    source = 1;
    sink = n;
    
    while true
        % 使用Bellman-Ford算法计算从源到每个节点的最短路径（按成本）
        dist = inf(1, n);
        pred = zeros(1, n);
        dist(source) = 0;
        
        for i = 1:n-1
            for u = 1:n
                for v = 1:n
                    if V(u, v) > flow(u, v) && dist(v) > dist(u) + C(u, v)
                        dist(v) = dist(u) + C(u, v);
                        pred(v) = u;
                    end
                end
            end
        end
        
        % 如果无法到达汇节点，则停止循环
        if dist(sink) == inf
            break;
        end
        
        % 计算增广路径上的可用容量
        pathCapacity = inf;
        v = sink;
        while v ~= source
            u = pred(v);
            pathCapacity = min(pathCapacity, V(u, v) - flow(u, v));
            v = u;
        end
        
        % 增加流量并更新总成本
        v = sink;
        while v ~= source
            u = pred(v);
            flow(u, v) = flow(u, v) + pathCapacity;
            flow(v, u) = flow(v, u) - pathCapacity;
            totalCost = totalCost + pathCapacity * C(u, v);
            v = u;
        end
    end
    wf = sum(flow(1,:));
    % 只显示正数的流
    positiveFlow = max(flow, 0);
    
    disp('数据流矩阵:');
    disp(positiveFlow);
    disp('汇节点此次收到的最大流：');
    disp(wf);
    disp(['该次传输最小消耗: ', num2str(totalCost)]);
    if wf==mf
    disp('此次生成,设计算法与matlab最大流计算结果相同');
else
    disp('此次生成，设计算法与matlab最大流计算结果不同');
    end
end




