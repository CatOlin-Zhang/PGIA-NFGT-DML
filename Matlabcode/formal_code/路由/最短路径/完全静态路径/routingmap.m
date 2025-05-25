% 创建一个示例路径矩阵
pathMatrix = [
    1 2 3 4 5;
    2 3 4 5 0;
    3 4 5 0 0;
    4 5 0 0 0;
    5 0 0 0 0
];

% 创建一个示例带宽矩阵
bandwidthMatrix = [
    0 50 0 0 0;
    50 0 65 0 0;
    0 65 0 70 0;
    0 0 70 0 80;
    0 0 0 80 0
];

% 调用函数生成流量矩阵
flowMatrix = routingProtocol(pathMatrix, bandwidthMatrix);

% 显示结果
disp('生成的流量矩阵:');
disp(flowMatrix);


function flowMatrix = routingProtocol(pathMatrix, bandwidthMatrix)
    % 获取路径矩阵和带宽矩阵的大小
    [pathRows, pathCols] = size(pathMatrix);
    [~, bwCols] = size(bandwidthMatrix);
    
    % 初始化流量矩阵
    flowMatrix = zeros(bwCols, bwCols);
    
    % 遍历每一条路径
    for i = 1:pathRows
        currentPath = pathMatrix(i, :);
        
        % 计算当前路径的有效长度
        validLength = sum(currentPath > 0);
        
        % 如果路径为空或只有一个节点，则跳过
        if validLength <= 1
            continue;
        end
        
        % 初始化最小带宽为无穷大
        minBandwidth = inf;
        
        % 检查路径中的每条边是否存在并计算最小带宽
        for j = 1:(validLength - 1)
            startNode = currentPath(j);
            endNode = currentPath(j + 1);
            
            % 如果起始节点或结束节点超出范围，跳过
            if startNode < 1 || startNode > bwCols || endNode < 1 || endNode > bwCols
                break;
            end
            
            % 获取当前边的带宽
            currentBandwidth = bandwidthMatrix(startNode, endNode);
            
            % 如果当前边不存在（带宽为0），跳过该路径
            if currentBandwidth == 0
                break;
            end
            
            % 更新最小带宽
            if currentBandwidth < minBandwidth
                minBandwidth = currentBandwidth;
            end
        end
        
        % 如果找到有效路径且最小带宽大于0，则扣除带宽并记录流量
        if minBandwidth > 0
            % 确定实际可分配的流量
            allocatedFlow = min(minBandwidth, 40);
            
            for j = 1:(validLength - 1)
                startNode = currentPath(j);
                endNode = currentPath(j + 1);
                
                % 扣除带宽
                bandwidthMatrix(startNode, endNode) = bandwidthMatrix(startNode, endNode) - allocatedFlow;
                
                % 记录流量
                flowMatrix(startNode, endNode) = flowMatrix(startNode, endNode) + allocatedFlow;
            end
        end
    end
    
    % 返回流量矩阵
end







