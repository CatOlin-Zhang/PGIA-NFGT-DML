function [flowMatrix,wf,point_in_flow_ratio,variance,total_costsum] = mincost_static(adjMatrix,cost_mat,data_per_round)
 n = size(adjMatrix, 1);   
 flowMatrix = mincost_static(adjMatrix, data_per_round) ;

%  disp(flowMatrix);
 
 
%最大流计算
wf=sum(flowMatrix(1,:));
flowarray = flowprocess(flowMatrix);
% disp(flowMatrix);
%汇聚比例计算
pointinflow = sum(flowarray(:) > 0);
 fprintf('静态最短路径汇聚比例%d / %d\n',pointinflow,n);
point_in_flow_ratio = pointinflow / n;
 
%消耗计算
   total_costMat = flowMatrix.*cost_mat;
   total_costsum = nansum(total_costMat(:));
   fprintf("静态最短路径传输消耗：%f\n",total_costsum );


%方差计算

    nonZeroElements = flowarray(flowarray>=0);
    variance = var(nonZeroElements) / pointinflow;
    fprintf("静态最短路径流矩阵方差：%f\n",variance);
   
    
    
   function flowarry = flowprocess(FlowMat)
            % 计算每行的行和
    rowSums = sum(FlowMat, 2);
    
    % 计算每列的列和
    colSums = sum(FlowMat, 1);
    
    % 初始化一个新的数组来存储结果
    flowarry = zeros(size(rowSums));
    
    % 计算行和减去列和并将结果存储在resultArray中
    for b = 1:size(FlowMat, 1)
        flowarry(b) = rowSums(b) - colSums(b);
    end
    end
   function [flowMatrix] = mincost_static(adjMatrix,data_per_round)
    % adjMatrix: 图的邻接矩阵
    % cost_mat: 成本矩阵
    % upper_bandwidth: 上限带宽矩阵
    % data_per_round: 每个节点每轮可以主动发出的最大数据量

    n = size(adjMatrix, 1); % 图中节点的数量
    lastNode = n; % 最后一个节点
    maxBandwidths = zeros(1, n); % 初始化最大带宽数组
    paths = cell(n, 1); % 存储路径
    flowMatrix = zeros(n, n); % 初始化流矩阵

    currentAdjMatrix = adjMatrix; 

    for startNode = 1:n-1 % 不需要从最后一个节点开始
        bandwidth = zeros(1, n); % 初始化带宽数组
        bandwidth(startNode) = inf; % 起始节点的带宽设为无穷大
        visited = false(1, n); % 记录节点是否被访问过
        parent = zeros(1, n); % 父节点数组用于记录路径

        u = startNode;
        while u ~= lastNode && ~all(visited)
            visited(u) = true;
            nextNode = find(currentAdjMatrix(u, :) > 0 & ~visited, 1);
            if isempty(nextNode)
                break; % 没有可用路径
            end
            parent(nextNode) = u;
            u = nextNode;
        end

        % 构建路径
        path = [];
        node = lastNode;
        while node ~= 0
            path = [node, path];
            node = parent(node);
        end
        paths{startNode} = path;

        % 处理节点到自身的特殊情况
        if length(path) == 1
            maxBandwidths(startNode) = 0;
        else
            % 找到路径上的最小带宽
            minPathBandwidth = inf;
            for j = 1:length(path)-1
                u = path(j);
                v = path(j+1);
                if currentAdjMatrix(u, v) < minPathBandwidth
                    minPathBandwidth = currentAdjMatrix(u, v);
                end
            end

            % 应用带宽上限和 data_per_round
            allocatedBandwidth = min(minPathBandwidth, data_per_round);

            % 更新最大可用带宽
            maxBandwidths(startNode) = allocatedBandwidth;

            % 更新流矩阵
            for j = 1:length(path)-1
                u = path(j);
                v = path(j+1);
                flowMatrix(u, v) = flowMatrix(u, v) + allocatedBandwidth;
            end

            % 扣除路径上的带宽
            for j = 1:length(path)-1
                u = path(j);
                v = path(j+1);
                currentAdjMatrix(u, v) = currentAdjMatrix(u, v) - allocatedBandwidth;
                currentAdjMatrix(v, u) = currentAdjMatrix(v, u) - allocatedBandwidth;
            end
        end
    end

% 输出每个节点到最后一个节点的最大可用带宽和路径
%     disp('每个节点到最后一个节点的最大可用带宽:');
%     for i = 1:n
%         fprintf('节点 %d 到 最后一个节点 的最大可用带宽: %.2f\n', i, maxBandwidths(i));
%         fprintf('路径: ');
%         disp(paths{i});
%     end
% 
%     % 输出流矩阵
%     disp('流矩阵:');
%     disp(flowMatrix);
end

end







