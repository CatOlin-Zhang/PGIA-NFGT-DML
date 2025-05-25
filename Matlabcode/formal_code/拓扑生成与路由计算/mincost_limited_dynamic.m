function [flowMatrix,wf,point_in_flow_ratio,variance,total_costsum] = mincost_limited_dynamic(adjMatrix,cost_mat,upper_bandwidth,onesRatio)
    % adjMatrix: 图的邻接矩阵

    n = size(adjMatrix, 1); % 图中节点的数量
  
    changedadjMatrix=static_routmap(n,onesRatio,adjMatrix);
    lastNode = n; % 最后一个节点
    maxBandwidths = zeros(1, n); % 初始化最大带宽数组
    paths = cell(n, 1); % 存储路径
    flowMatrix = zeros(n, n); % 初始化流矩阵

    currentAdjMatrix = changedadjMatrix; 

    for startNode = 1:n
        bandwidth = zeros(1, n); % 初始化带宽数组
        bandwidth(startNode) = inf; % 起始节点的带宽设为无穷大
        visited = false(1, n); % 记录节点是否被访问过
        parent = zeros(1, n); % 父节点数组用于记录路径

        for i = 1:n-1
            
            [~, maxIndex] = max(bandwidth .* ~visited);
            u = maxIndex;
            visited(u) = true;

            % 更新相邻节点的带宽和父节点
            for v = 1:n
                if ~visited(v) && currentAdjMatrix(u, v) > 0
                    newBandwidth = min(bandwidth(u), currentAdjMatrix(u, v));
                    if newBandwidth > bandwidth(v)
                        bandwidth(v) = newBandwidth;
                        parent(v) = u;
                    end
                end
            end
        end

        % 获取最后一个节点的索引
        maxBandwidths(startNode) = bandwidth(lastNode);

        % 构建路径
        path = [];
        node = lastNode;
        while node ~= 0
            path = [node, path];
            node = parent(node);
        end
        paths{startNode} = path;

       
        if length(path) == 1
            maxBandwidths(startNode) = 0;
        else
            % 更新流矩阵
            for j = 1:length(path)-1
                u = path(j);
                v = path(j+1);
                flowMatrix(u, v) = flowMatrix(u, v) + maxBandwidths(startNode);
            end

            % 扣除路径上的带宽
            for j = 1:length(path)-1
                u = path(j);
                v = path(j+1);
                currentAdjMatrix(u, v) = currentAdjMatrix(u, v) - maxBandwidths(startNode);
                currentAdjMatrix(v, u) = currentAdjMatrix(v, u) - maxBandwidths(startNode);
            end
        end
    end

% 调试用：输出每个节点到最后一个节点的最大可用带宽和路径
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

%最大流计算
wf=sum(flowMatrix(1,:));

%汇聚比例计算
   pointinflow = sum(flowMatrix(1,:) ~= 0);
 fprintf('静态最短路径汇聚比例%d / %d\n',pointinflow,n-2);
 point_in_flow_ratio=pointinflow/(n-2);
 
%消耗计算
   total_costMat = flowMatrix.*cost_mat;
   total_costsum = nansum(total_costMat(:));
   fprintf("静态最短路径传输消耗：%f\n",total_costsum );


%方差计算
   flow_varmatrix=(upper_bandwidth-(adjMatrix-flowMatrix));
    flowvar=flow_varmatrix(1,:);
    nonZeroElements = flowvar(flowvar~=0);
    variance=var(nonZeroElements)/pointinflow;
    fprintf("静态最短路径流矩阵方差：%f\n",variance);
    function symmetricMatrix = static_routmap(size, onesRatio,adjMatrix)
totalElements = size * size;
numOnes = round(totalElements * onesRatio);

% 创建一个包含numOnes个1和totalElements-numOnes个0的向量
binaryVector = [ones(1, numOnes), zeros(1, totalElements - numOnes)];

% 随机打乱这个向量
shuffledVector = randperm(length(binaryVector));
binaryVector = binaryVector(shuffledVector);

% 将向量重塑成size x size的方阵
upperTriangularMatrix = triu(reshape(binaryVector, size, size));

% 生成对称矩阵
symmetricMatrix = upperTriangularMatrix + upperTriangularMatrix';

% 确保对角线上的元素不超过1
symmetricMatrix(logical(symmetricMatrix > 1)) = 1;
symmetricMatrix=symmetricMatrix.*adjMatrix;
disp(symmetricMatrix);
end
end







