function matrix = add_virtual_end_node_forV(matrix, maxflowcanreceive)
    % 假设 adjMatrix 是现有的邻接矩阵
    adjMatrix = matrix;

    % 计算每一行的和
    rowSums = sum(adjMatrix, 2);
    
    % 计算每一列的和，并将其转换为列向量
    colSums = sum(adjMatrix, 1)';
    
    % 合并行和和列和，并找到最大值及其索引
    combinedSums = [rowSums; colSums];
    [maxValue, linearIndex] = max(combinedSums(:));

    % 如果最大值来自行和，则确定对应的行索引
    if linearIndex <= length(rowSums)
        nodeIndex = linearIndex;
    else
        % 如果最大值来自列和，则确定对应的列索引
        nodeIndex = linearIndex - length(rowSums);
    end

    % 创建新的邻接矩阵
    n = size(adjMatrix, 1); % 原始矩阵的大小
    newAdjMatrix = zeros(n+1, n+1); % 新矩阵的大小

    % 复制原始矩阵的内容到新矩阵
    newAdjMatrix(1:n, 1:n) = adjMatrix;

    % 设置新节点与最大带宽点之间的连接
    newAdjMatrix(nodeIndex, n+1) = maxflowcanreceive;
    newAdjMatrix(n+1, nodeIndex) = maxflowcanreceive;

    matrix = newAdjMatrix;
end



