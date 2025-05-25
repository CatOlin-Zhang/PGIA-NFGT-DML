%�����·���㷨:�����������󣬴��۾��󣬴�������;���������������ֵ�������ģ���۽ڵ��������������������
function [flow_matrix,max_flow,total_costsum,point_in_flow_ratio,variance] = Maxflowrouting(capacity,costMatrix,upper_bandwidth)
    num_vertices = size(capacity, 1);
    source = 1; % Դ��̶�Ϊ���Ϊ1�ĵ�
    sink = num_vertices; % ���̶�Ϊ������ĵ�
    
    residual_capacity = capacity;
    flow_matrix = zeros(num_vertices);

    while true
        parent = bfs(residual_capacity, source, sink);
        
        if parent(sink) == 0
            break; % ���û���ҵ�����·�������˳�ѭ��
        end
        
        path_flow = inf;
        v = sink;
        while v ~= source
            u = parent(v);
            path_flow = min(path_flow, residual_capacity(u, v));
            v = u;
        end
        
        v = sink;
        while v ~= source
            u = parent(v);
            residual_capacity(u, v) = residual_capacity(u, v) - path_flow;
            residual_capacity(v, u) = residual_capacity(v, u) + path_flow;
            flow_matrix(u, v) = flow_matrix(u, v) + path_flow;
            v = u;
        end
        
        max_flow = sum(flow_matrix(1, :));
    end
   % disp(flow_matrix);
   total_costMat = flow_matrix.*costMatrix;
   total_costsum = sum(total_costMat(:));
   fprintf("������������ģ�%f\n",total_costsum );
  
   pointinflow = sum(flow_matrix(1,:) ~= 0);
   fprintf('�������۱�����%d / %d\n',pointinflow,num_vertices-2)
   %���ڷ����������ӦΪ��ǰ�ܺ�
   %�൱�ڴ�������-����������-������:���������൱�ڳ�ʼʣ�����ȥ��������
   flow_varmatrix=(upper_bandwidth-(capacity-flow_matrix));
    flowvar=flow_varmatrix(1,:);
    nonZeroElements = flowvar(flowvar~=0);
    variance=var(nonZeroElements)/pointinflow;
    point_in_flow_ratio=pointinflow/(num_vertices-2);
    fprintf("����������󷽲%f\n",variance);
end

function parent = bfs(residual_capacity, source, sink)
    num_vertices = size(residual_capacity, 1);
    visited = false(1, num_vertices);
    queue = [];
    
    parent = zeros(1, num_vertices); % �� parent ����Ϊ��ֵ����
    
    visited(source) = true;
    queue(end+1) = source;
    
    while ~isempty(queue)
        u = queue(1);
        queue(1) = [];
        
        for v = 1:num_vertices
            if ~visited(v) && residual_capacity(u, v) > 0
                queue(end+1) = v;
                visited(v) = true;
                parent(v) = u; % ʹ����ֵ������ֵ
                
                if v == sink
                    return;
                end
            end
        end
    end
end