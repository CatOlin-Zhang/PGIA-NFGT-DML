function [......
    time_spfa, time_greedy, time_dynamic, time_static,......
    spfa_cost, greedy_cost, dynamic_cost, static_cost,......
    var_spfa, var_greedy, var_dynamic, var_static,......
    pointinflow_ratio_spfa, pointinflow_ratio_greedy, pointinflow_ratio_dynamic, pointinflow_ratio_static, ......
    maxflow_spfa, maxflow_greedy, maxflow_dynamic, maxflow_static ......
    ] = maincode()
    % ��������

    clc;
    clear;
    
    % ���ò���
    N = 30; % �ڵ���
    pe = 0.5; % �����ɸ���
    min_bandwidth = 10; % ��С����
    max_bandwidth = 100; % ������
    data_per_round = 50; % ÿ�ֵ����ڵ������͵������е����������
    maxflowcanreceive = Inf; % ÿ�ֻ�۽ڵ�����������н��յ����������
    generated_spare_ratio = 0.5; % ���������
    difference_of_bandwidth = 40; % ���ɱ�׼��

    % �����������ͼ�����������ȡ�ڽӾ���ʹ��۾���
    [G, ~, adjacencyMatrix, costMatrix] = generate_random_topology_and_bandwidth(N, pe, min_bandwidth, max_bandwidth, generated_spare_ratio, difference_of_bandwidth);
    
    %����̬·�ɲ���Ҫ�����ڵ�
    adjacencyMatrix_for_dynamic = adjacencyMatrix;
    adjacencyMatrix_for_static = adjacencyMatrix;
    costMatrix_for_dynamic = costMatrix;
    costMatrix_for_static = costMatrix;
    
    
    if N <= 10
        disp('�ڽӾ���:');
        disp(adjacencyMatrix);
        disp('���۾���:');
        disp(costMatrix);
    else
        
    end

    % ������Ŀ��
    adjacencyMatrix = add_virtual_end_nodeforV(adjacencyMatrix, maxflowcanreceive);
    costMatrix = add_virtual_end_nodeforC(adjacencyMatrix, costMatrix);

    % ������Դ
    adjacencyMatrix = add_virtual_souce_nodeforV(adjacencyMatrix, data_per_round);
    costMatrix = add_virtual_souce_nodeforC(costMatrix);

    % ��ʾ���
    if N <= 10
        disp('�������ڵ����ڽӾ���:');
        disp(adjacencyMatrix);
        disp('�������ڵ��Ĵ��۾���:');
        disp(costMatrix);
    else
        
    end

    % SPFA·�ɼ���
    [flow_spfa,spfa_cost,~,pointinflow_ratio_spfa,var_spfa, maxflow_spfa] = SPFArouting(adjacencyMatrix, costMatrix,max_bandwidth);
    % ʱ�����
    time_spfa = timecalculate_forSPFAwithMaxflow(flow_spfa, data_per_round);
    fprintf("SPFA·�ɴ���ʱ�䣺%f\n", time_spfa);

    % �����·�ɼ���
    [flow_greedy,maxflow_greedy, greedy_cost,pointinflow_ratio_greedy,var_greedy] = Maxflowrouting(adjacencyMatrix, costMatrix, max_bandwidth);
    % ʱ�����
    time_greedy = timecalculate_forSPFAwithMaxflow(flow_greedy, data_per_round);
    fprintf("�����·�ɴ���ʱ�䣺%f\n", time_greedy);

    %��̬���·������
    [flow_dynamic,maxflow_dynamic,pointinflow_ratio_dynamic,var_dynamic,dynamic_cost]= mincost_dynamic(adjacencyMatrix_for_dynamic,costMatrix_for_dynamic,max_bandwidth, data_per_round);
    %ʱ�����
    time_dynamic = timecalculate_forDynamicwithStatic(flow_dynamic, data_per_round);
    fprintf("��̬���·��·�ɴ���ʱ�䣺%f\n", time_dynamic);
    
    %��̬���·������
    [flow_static,maxflow_static,pointinflow_ratio_static,var_static,static_cost]= mincost_static(adjacencyMatrix_for_static,costMatrix_for_static, data_per_round);
    %ʱ�����
    time_static = timecalculate_forDynamicwithStatic(flow_static, data_per_round);
    fprintf("��̬���·��·�ɴ���ʱ�䣺%f\n", time_static);
    
    % ��ʾ���
    if N <= 10
        figure;
        plot(G, 'Layout', 'force', 'LineWidth', 2);
        title('�������ͼ');
    else
        
    end
%     disp(maxflow_spfa );
%     disp(maxflow_greedy);
%     disp(maxflow_dynamic);
end



