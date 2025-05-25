function matrix = add_virtual_souce_nodeforV(matrix,maxflow_per_round)
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