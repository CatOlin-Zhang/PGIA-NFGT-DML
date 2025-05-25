capacity =[
    0     5     8     0     0;
    0     0     0     3     4;
    0     2     0    10     0;
    0     0     0     0     8;
    0     0     0     0     0
];

[max_flow, flow_matrix] = edmonds_karp(capacity);
disp(['Maximum Flow: ', num2str(max_flow)]);
disp('Flow Matrix:');
disp(flow_matrix);

function [max_flow, flow_matrix] = edmonds_karp(capacity)
    % capacity: 邻接矩阵表示的图，元素为边的容量
    
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
        
        max_flow = sum(flow_matrix(source, :));
    end
    
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