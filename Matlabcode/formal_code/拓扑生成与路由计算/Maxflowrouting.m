%最大流路由算法:传入容量矩阵，代价矩阵，带宽上限;返回流矩阵，最大流值，总消耗，汇聚节点比例，最终数据流方差
function [flow_matrix,max_flow,total_costsum,point_in_flow_ratio,variance] = Maxflowrouting(capacity,costMatrix,upper_bandwidth)
    num_vertices = size(capacity, 1);
    source = 1; % 源点固定为编号为1的点
    sink = num_vertices; % 汇点固定为编号最大的点
    
    residual_capacity = capacity;
    flow_matrix = zeros(num_vertices);

    while true
        parent = bfs(residual_capacity, source, sink);
        
        if parent(sink) == 0
            break; % 如果没有找到增广路径，则退出循环
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
   fprintf("最大流传输消耗：%f\n",total_costsum );
  
   pointinflow = sum(flow_matrix(1,:) ~= 0);
   fprintf('最大流汇聚比例：%d / %d\n',pointinflow,num_vertices-2)
   %用于方差的流计算应为当前总和
   %相当于带宽上限-（容量矩阵-流矩阵）:容量矩阵相当于初始剩余带宽，去除流消耗
   flow_varmatrix=(upper_bandwidth-(capacity-flow_matrix));
    flowvar=flow_varmatrix(1,:);
    nonZeroElements = flowvar(flowvar~=0);
    variance=var(nonZeroElements)/pointinflow;
    point_in_flow_ratio=pointinflow/(num_vertices-2);
    fprintf("最大流流矩阵方差：%f\n",variance);
end

function parent = bfs(residual_capacity, source, sink)
    num_vertices = size(residual_capacity, 1);
    visited = false(1, num_vertices);
    queue = [];
    
    parent = zeros(1, num_vertices); % 将 parent 定义为数值数组
    
    visited(source) = true;
    queue(end+1) = source;
    
    while ~isempty(queue)
        u = queue(1);
        queue(1) = [];
        
        for v = 1:num_vertices
            if ~visited(v) && residual_capacity(u, v) > 0
                queue(end+1) = v;
                visited(v) = true;
                parent(v) = u; % 使用数值索引赋值
                
                if v == sink
                    return;
                end
            end
        end
    end
end