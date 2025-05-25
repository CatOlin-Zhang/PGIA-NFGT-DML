function  matrix = add_virtual_end_nodeforC(V, C)

    cost_value=1e-15;
    n=size(C,1);
    % 创建新的代价矩阵 C_new
    new_C = zeros(n+1);
    new_C(1:n, 1:n) = C;
    
    % 找到 V 矩阵最后一行或最后一列中的非零元素位置
    last_row_nonzero_col = find(V(end, :) ~= 0);
    last_col_nonzero_row = find(V(:, end) ~= 0);
    
    if isempty(last_row_nonzero_col) && isempty(last_col_nonzero_row)
        error('容量矩阵错误，无法建立到目标节点的连接');
    elseif ~isempty(last_row_nonzero_col)
        % 如果在最后一行找到非零元素
        col_index = last_row_nonzero_col(1);
        new_C(end, col_index) = cost_value; 
        fprintf("此次找到最大带宽点:%d\n",col_index)
    elseif ~isempty(last_col_nonzero_row)
        % 如果在最后一列找到非零元素
        row_index = last_col_nonzero_row(1);
        new_C(row_index, end) = cost_value; 
        fprintf("此次找到最大带宽点:%d\n",col_index)
    end
        matrix=new_C;
end







