clc;
clear;
% 设置参数
N = 7; % 节点数 一般大于7，较少节点时生成函数性能不稳定，不建议大于15，因为路由计算量较大。
pe = 0.3; % 边生成概率
min_bandwidth = 10; % 最小带宽
max_bandwidth = 100; % 最大带宽
maxflowperround = 40;%
%更改节点数后需要去45行修改matlab计算节点编号（默认为5）（路由只能计算1到numNode）。

% 生成随机拓扑图及其带宽，并获取邻接矩阵和代价矩阵
[G, bandwidths, V, C] = generate_random_topology_and_bandwidth(N, pe, min_bandwidth, max_bandwidth);

% 显示结果
disp('原始邻接矩阵:');
disp(V);
disp('原始代价矩阵:');
disp(C);

figure;
plot(G, 'Layout', 'force', 'LineWidth', 2);
title('随机拓扑图');
%% 添加虚拟节点
V = add_virtual_nodeforV(V,maxflowperround);
C = add_virtual_nodeforC(C);

% 显示结果
disp('添加虚拟节点后的邻接矩阵:');
disp(V);
disp('添加虚拟节点后的代价矩阵:');
disp(C);
%% 创建适合matlab_maxflow()函数的计算环境
% 获取矩阵的大小
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
% 计算从节点 1 到节点 5 的最大流
mf = maxflow(G, 1, 7);
disp('Matlab最大流计算结果：');
disp(mf);
n=size(V,2);

%% 算法数据初始化
wf=0;wf0=Inf; %wf:最大流量, wf0:预设最大流量，这里初始化为无限大表示不限制
f = zeros(n,n); %初始化流量矩阵
while 1
%% 加权网络
    a = inf*(ones(n,n)-eye(n,n)); %有向加权图
    for i=1:n
        for j=1:n
            if V(i,j)>0&&f(i,j)==0
                a(i,j)=C(i,j);
            elseif V(i,j)>0&&f(i,j)==V(i,j)
                a(j,i)=-C(i,j);
            elseif V(i,j)>0
                a(i,j)=C(i,j);a(j,i)=-C(i,j);
            end
        end
    end
    
    %% 查找最短路径使用Bellman-Ford算法
    p = inf * ones(1,n); %初始化距离向量
    p(1) = 0;
    s = 1:n;
    s(1) = 0;
    for k=1:n
        pd=1; %标记是否更新了最短路径
        for i=2:n
            for j=1:n
                if p(i)>p(j)+a(j,i)
                    p(i)=p(j)+a(j,i);
                    s(i)=j;
                    pd=0;
                end
            end
        end
        if pd
            break;
        end
    end
    if p(n)==Inf
        disp('计算完成');
        break;
    end %如果没有找到从起点到终点的可行路径，结束循环

    %% 流量调整
    k=n;
    dvt=Inf;%dvt:可调整的最大流量
    t=n; 
    while 1 %计算可调整的最大流量
        if a(s(t),t)>0
            dvtt=V(s(t),t)-f(s(t),t); %正向弧
        elseif a(s(t),t)<0
            dvtt=f(t,s(t)); %反向弧
        end 
        if dvt>dvtt
            dvt=dvtt;
        end
        if s(t)==1
            break;
        end %当到达起点时结束
        t=s(t);
    end
    pd=0;
    if wf+dvt>=wf0 %如果当前总流量加上可调整的最大流量超过了预设的最大值
        dvt=wf0-wf;
        pd=1;
    end
    t=n;
    while 1 %调整流量
        if a(s(t),t)>0
            f(s(t),t)=f(s(t),t)+dvt; %正向弧
        elseif a(s(t),t)<0
            f(t,s(t))=f(t,s(t))-dvt; %反向弧
        end 
        if s(t)==1
            break;
        end %当到达起点时结束
        t=s(t);
    end
    if pd
        break;
    end %如果达到预设的最大流量，结束循环
end

%% 显示计算结果与结果验证
wf = sum(f(1,:));
zwf = sum(sum(C.*f)); %计算最小损失
disp('数据流矩阵')
disp(f); %流矩阵
disp('汇节点接收到的最大流量')
disp(wf);%最大流
disp('该次传输的最小消耗')
disp(zwf); %最小损失，为数据流矩阵对应单位数据量与代价矩阵对应项相乘。
if wf==mf
    disp('此次生成,设计算法与matlab最大流计算结果相同');
else
    disp('此次生成,设计算法与matlab最大流计算结果不同');
end


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
        if (forwardBandwidth > 0 && forwardBandwidth~=max_bandwidth) %防止costNaN问题生成下同
            costMatrix(startNode, endNode) = 1 / (max_bandwidth - forwardBandwidth);
        else
            costMatrix(startNode, endNode) = 0; % 设置为0
        end
        if (backwardBandwidth > 0 && backwardBandwidth~=max_bandwidth)
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
function matrix = add_virtual_nodeforV(matrix,maxflow_per_round)
    % 在矩阵的第一行和第一列添加新的节点
    n = size(matrix, 1);
    new_matrix = zeros(n+1);
    
    % 将原矩阵复制到新矩阵中
    new_matrix(2:end, 2:end) = matrix;
    
    % 设置虚拟节点与其他节点的连接关系
    for i = 2:n+1
        if i ~= n+1
            new_matrix(1, i) = maxflow_per_round;   %替代原始次数的限制
            new_matrix(i, 1) = maxflow_per_round;
        end
    end
    
    matrix = new_matrix;
end
function matrix = add_virtual_nodeforC(matrix)
    % 在矩阵的第一行和第一列添加新的节点
    n = size(matrix, 1);
    new_matrix = zeros(n+1);
    
    % 将原矩阵复制到新矩阵中
    new_matrix(2:end, 2:end) = matrix;
    
    % 设置虚拟节点与其他节点的连接关系
    for i = 2:n+1
        if i ~= n+1
            new_matrix(1, i) = 1e-15;   %替代原始次数的限制
            new_matrix(i, 1) = 1e-15;
        end
    end
    matrix = new_matrix;
end