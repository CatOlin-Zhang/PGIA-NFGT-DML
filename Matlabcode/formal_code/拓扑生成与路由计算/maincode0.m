%联合运行
clc;
clear;

% 设置参数
N = 7; % 节点数
pe = 0.5; % 边生成概率
min_bandwidth = 10; % 最小带宽
max_bandwidth = 100; % 最大带宽
data_per_round = 40;%每轮单个节点允许发送到网络中的最大数据量
maxflowcanreceive=Inf;%每轮汇聚节点允许从网络中接收的最大数据量
generated_spare_ratio=0.5;%网络空闲率
difference_of_bandwidth=10;%生成标准差



% 生成随机拓扑图及其带宽，并获取邻接矩阵和代价矩阵

[G, bandwidths, adjacencyMatrix, costMatrix] = generate_random_topology_and_bandwidth(N, pe, min_bandwidth, max_bandwidth,generated_spare_ratio,difference_of_bandwidth);

if N<=10
    disp('邻接矩阵:');
    disp(adjacencyMatrix);
    disp('代价矩阵:');
    disp(costMatrix);
else
    
end


%加虚拟目的
adjacencyMatrix=add_virtual_end_nodeforV(adjacencyMatrix,maxflowcanreceive);
costMatrix=add_virtual_end_nodeforC(adjacencyMatrix,costMatrix);

%加虚拟源
adjacencyMatrix=add_virtual_souce_nodeforV(adjacencyMatrix,data_per_round);
costMatrix=add_virtual_souce_nodeforC(costMatrix);


% 显示结果
if N<=10
disp('添加虚拟节点后的邻接矩阵:');
 disp(adjacencyMatrix);
 disp('添加虚拟节点后的代价矩阵:');
 disp(costMatrix);
else
    
    
end
%SPFA路由计算
flow=SPFArouting(adjacencyMatrix,costMatrix,max_bandwidth);
%时间计算
time=timecalculate(flow,data_per_round);
fprintf("SPFA路由传输时间：%f\n",time);

%贪心最大流路由计算
[flow,maxflow_of_MAX]=Maxflowrouting(adjacencyMatrix,costMatrix,max_bandwidth);
%时间计算
time=timecalculate(flow,data_per_round);
fprintf("贪心最大流路由传输时间：%f\n",time);


% 显示结果
if N<=10
figure;
plot(G, 'Layout', 'force', 'LineWidth', 2);
title('随机拓扑图');
else
    
end
