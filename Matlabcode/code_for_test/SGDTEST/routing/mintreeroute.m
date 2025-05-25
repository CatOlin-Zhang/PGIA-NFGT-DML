clc;
clear;

% 图的邻接矩阵
adjacencyMatrix = [
    0, 2, 0, 6, 0, 4, 1, 0, 0, 3;
    2, 0, 3, 8, 5, 1, 0, 2, 0, 0;
    0, 3, 0, 0, 7, 0, 0, 3, 2, 0;
    6, 8, 0, 0, 9, 0, 0, 4, 0, 0;
    0, 5, 7, 9, 0, 0, 0, 0, 0, 0;
    4, 1, 0, 0, 0, 0, 2, 0, 1, 0;
    1, 0, 0, 0, 0, 2, 0, 3, 0, 1;
    0, 2, 3, 4, 0, 0, 3, 0, 2, 0;
    0, 0, 2, 0, 0, 1, 0, 2, 0, 4;
    3, 0, 0, 0, 0, 0, 1, 0, 4, 0
];

% 判断是否为稠密图
numVertices = size(adjacencyMatrix, 1);
numEdges = nnz(triu(adjacencyMatrix)); % 计算上三角非零元素的数量
density = (2 * numEdges) / (numVertices * (numVertices - 1));

if density > 0.5
    fprintf('图是稠密图，使用Kruskal算法。\n');
    mst = kruskalAlgorithm(adjacencyMatrix);
else
    fprintf('图是稀疏图，使用Prim算法。\n');%1
    mst = primAlgorithm(adjacencyMatrix);
end

disp('最小生成树的邻接矩阵:');
disp(mst);

% 绘制原始图和最小生成树
figure;
subplot(1, 2, 1);
plotGraph(adjacencyMatrix, '原始图');

subplot(1, 2, 2);
plotGraph(mst, '最小生成树');

% Prim算法实现
function mst = primAlgorithm(adjMatrix)
    n = size(adjMatrix, 1);
    visited = false(n, 1);
    mst = zeros(n);

    startNode = 1; % 从第一个节点开始
    visited(startNode) = true;

    while sum(visited) < n
        minWeight = inf;
        u = 0;
        v = 0;

        for i = 1:n
            if visited(i)
                for j = 1:n
                    if ~visited(j) && adjMatrix(i, j) ~= 0 && adjMatrix(i, j) < minWeight
                        minWeight = adjMatrix(i, j);
                        u = i;
                        v = j;
                    end
                end
            end
        end

        if u ~= 0 && v ~= 0
            mst(u, v) = minWeight;
            mst(v, u) = minWeight;
            visited(v) = true;
        end
    end
end

% Kruskal算法实现
function mst = kruskalAlgorithm(adjMatrix)
    n = size(adjMatrix, 1);
    edges = [];
    for i = 1:n-1
        for j = i+1:n
            if adjMatrix(i, j) ~= 0
                edges = [edges; i, j, adjMatrix(i, j)];
            end
        end
    end

    % 按权重排序边
    edges = sortrows(edges, 3);

    parent = 1:n;
    rank = zeros(n, 1);
    mst = zeros(n);

    edgeCount = 0;
    for i = 1:size(edges, 1)
        u = edges(i, 1);
        v = edges(i, 2);
        weight = edges(i, 3);

        rootU = findSet(parent, u);
        rootV = findSet(parent, v);

        if rootU ~= rootV
            unionSets(parent, rank, rootU, rootV);
            mst(u, v) = weight;
            mst(v, u) = weight;
            edgeCount = edgeCount + 1;
            if edgeCount == n - 1
                break;
            end
        end
    end
end

% 查找集合根节点
function root = findSet(parent, node)
    if parent(node) ~= node
        parent(node) = findSet(parent, parent(node));
    end
    root = parent(node);
end

% 合并两个集合
function unionSets(parent, rank, root1, root2)
    if rank(root1) < rank(root2)
        parent(root1) = root2;
    elseif rank(root1) > rank(root2)
        parent(root2) = root1;
    else
        parent(root2) = root1;
        rank(root1) = rank(root1) + 1;
    end
end

% 绘制图函数
function plotGraph(adjMatrix, titleText)
    G = graph(adjMatrix);
    p = plot(G);
    p.NodeLabel = string(1:numnodes(G));
    title(titleText);
end



