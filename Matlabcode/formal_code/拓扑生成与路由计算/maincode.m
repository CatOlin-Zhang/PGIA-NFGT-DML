function [......
    time_spfa, time_greedy, time_dynamic, time_static,......
    spfa_cost, greedy_cost, dynamic_cost, static_cost,......
    var_spfa, var_greedy, var_dynamic, var_static,......
    pointinflow_ratio_spfa, pointinflow_ratio_greedy, pointinflow_ratio_dynamic, pointinflow_ratio_static, ......
    maxflow_spfa, maxflow_greedy, maxflow_dynamic, maxflow_static ......
    ] = maincode()
    % 联合运行

    clc;
    clear;
    
    % 设置参数
    N = 30; % 节点数
    pe = 0.5; % 边生成概率
    min_bandwidth = 10; % 最小带宽
    max_bandwidth = 100; % 最大带宽
    data_per_round = 50; % 每轮单个节点允许发送到网络中的最大数据量
    maxflowcanreceive = Inf; % 每轮汇聚节点允许从网络中接收的最大数据量
    generated_spare_ratio = 0.5; % 网络空闲率
    difference_of_bandwidth = 40; % 生成标准差

    % 生成随机拓扑图及其带宽，并获取邻接矩阵和代价矩阵
    [G, ~, adjacencyMatrix, costMatrix] = generate_random_topology_and_bandwidth(N, pe, min_bandwidth, max_bandwidth, generated_spare_ratio, difference_of_bandwidth);
    
    %动静态路由不需要辅助节点
    adjacencyMatrix_for_dynamic = adjacencyMatrix;
    adjacencyMatrix_for_static = adjacencyMatrix;
    costMatrix_for_dynamic = costMatrix;
    costMatrix_for_static = costMatrix;
    
    
    if N <= 10
        disp('邻接矩阵:');
        disp(adjacencyMatrix);
        disp('代价矩阵:');
        disp(costMatrix);
    else
        
    end

    % 加虚拟目的
    adjacencyMatrix = add_virtual_end_nodeforV(adjacencyMatrix, maxflowcanreceive);
    costMatrix = add_virtual_end_nodeforC(adjacencyMatrix, costMatrix);

    % 加虚拟源
    adjacencyMatrix = add_virtual_souce_nodeforV(adjacencyMatrix, data_per_round);
    costMatrix = add_virtual_souce_nodeforC(costMatrix);

    % 显示结果
    if N <= 10
        disp('添加虚拟节点后的邻接矩阵:');
        disp(adjacencyMatrix);
        disp('添加虚拟节点后的代价矩阵:');
        disp(costMatrix);
    else
        
    end

    % SPFA路由计算
    [flow_spfa,spfa_cost,~,pointinflow_ratio_spfa,var_spfa, maxflow_spfa] = SPFArouting(adjacencyMatrix, costMatrix,max_bandwidth);
    % 时间计算
    time_spfa = timecalculate_forSPFAwithMaxflow(flow_spfa, data_per_round);
    fprintf("SPFA路由传输时间：%f\n", time_spfa);

    % 最大流路由计算
    [flow_greedy,maxflow_greedy, greedy_cost,pointinflow_ratio_greedy,var_greedy] = Maxflowrouting(adjacencyMatrix, costMatrix, max_bandwidth);
    % 时间计算
    time_greedy = timecalculate_forSPFAwithMaxflow(flow_greedy, data_per_round);
    fprintf("最大流路由传输时间：%f\n", time_greedy);

    %动态最短路径计算
    [flow_dynamic,maxflow_dynamic,pointinflow_ratio_dynamic,var_dynamic,dynamic_cost]= mincost_dynamic(adjacencyMatrix_for_dynamic,costMatrix_for_dynamic,max_bandwidth, data_per_round);
    %时间计算
    time_dynamic = timecalculate_forDynamicwithStatic(flow_dynamic, data_per_round);
    fprintf("动态最短路径路由传输时间：%f\n", time_dynamic);
    
    %静态最短路径计算
    [flow_static,maxflow_static,pointinflow_ratio_static,var_static,static_cost]= mincost_static(adjacencyMatrix_for_static,costMatrix_for_static, data_per_round);
    %时间计算
    time_static = timecalculate_forDynamicwithStatic(flow_static, data_per_round);
    fprintf("静态最短路径路由传输时间：%f\n", time_static);
    
    % 显示结果
    if N <= 10
        figure;
        plot(G, 'Layout', 'force', 'LineWidth', 2);
        title('随机拓扑图');
    else
        
    end
%     disp(maxflow_spfa );
%     disp(maxflow_greedy);
%     disp(maxflow_dynamic);
end



