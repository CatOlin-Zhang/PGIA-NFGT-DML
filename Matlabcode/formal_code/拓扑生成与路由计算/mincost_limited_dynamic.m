function [flowMatrix,wf,point_in_flow_ratio,variance,total_costsum] = mincost_limited_dynamic(adjMatrix,cost_mat,upper_bandwidth,onesRatio)
    % adjMatrix: ͼ���ڽӾ���

    n = size(adjMatrix, 1); % ͼ�нڵ������
  
    changedadjMatrix=static_routmap(n,onesRatio,adjMatrix);
    lastNode = n; % ���һ���ڵ�
    maxBandwidths = zeros(1, n); % ��ʼ������������
    paths = cell(n, 1); % �洢·��
    flowMatrix = zeros(n, n); % ��ʼ��������

    currentAdjMatrix = changedadjMatrix; 

    for startNode = 1:n
        bandwidth = zeros(1, n); % ��ʼ����������
        bandwidth(startNode) = inf; % ��ʼ�ڵ�Ĵ�����Ϊ�����
        visited = false(1, n); % ��¼�ڵ��Ƿ񱻷��ʹ�
        parent = zeros(1, n); % ���ڵ��������ڼ�¼·��

        for i = 1:n-1
            
            [~, maxIndex] = max(bandwidth .* ~visited);
            u = maxIndex;
            visited(u) = true;

            % �������ڽڵ�Ĵ���͸��ڵ�
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

        % ��ȡ���һ���ڵ������
        maxBandwidths(startNode) = bandwidth(lastNode);

        % ����·��
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
            % ����������
            for j = 1:length(path)-1
                u = path(j);
                v = path(j+1);
                flowMatrix(u, v) = flowMatrix(u, v) + maxBandwidths(startNode);
            end

            % �۳�·���ϵĴ���
            for j = 1:length(path)-1
                u = path(j);
                v = path(j+1);
                currentAdjMatrix(u, v) = currentAdjMatrix(u, v) - maxBandwidths(startNode);
                currentAdjMatrix(v, u) = currentAdjMatrix(v, u) - maxBandwidths(startNode);
            end
        end
    end

% �����ã����ÿ���ڵ㵽���һ���ڵ�������ô����·��
%     disp('ÿ���ڵ㵽���һ���ڵ�������ô���:');
%     for i = 1:n
%         fprintf('�ڵ� %d �� ���һ���ڵ� �������ô���: %.2f\n', i, maxBandwidths(i));
%         fprintf('·��: ');
%         disp(paths{i});
%     end
% 
%     % ���������
%     disp('������:');
%     disp(flowMatrix);

%���������
wf=sum(flowMatrix(1,:));

%��۱�������
   pointinflow = sum(flowMatrix(1,:) ~= 0);
 fprintf('��̬���·����۱���%d / %d\n',pointinflow,n-2);
 point_in_flow_ratio=pointinflow/(n-2);
 
%���ļ���
   total_costMat = flowMatrix.*cost_mat;
   total_costsum = nansum(total_costMat(:));
   fprintf("��̬���·���������ģ�%f\n",total_costsum );


%�������
   flow_varmatrix=(upper_bandwidth-(adjMatrix-flowMatrix));
    flowvar=flow_varmatrix(1,:);
    nonZeroElements = flowvar(flowvar~=0);
    variance=var(nonZeroElements)/pointinflow;
    fprintf("��̬���·�������󷽲%f\n",variance);
    function symmetricMatrix = static_routmap(size, onesRatio,adjMatrix)
totalElements = size * size;
numOnes = round(totalElements * onesRatio);

% ����һ������numOnes��1��totalElements-numOnes��0������
binaryVector = [ones(1, numOnes), zeros(1, totalElements - numOnes)];

% ��������������
shuffledVector = randperm(length(binaryVector));
binaryVector = binaryVector(shuffledVector);

% ���������ܳ�size x size�ķ���
upperTriangularMatrix = triu(reshape(binaryVector, size, size));

% ���ɶԳƾ���
symmetricMatrix = upperTriangularMatrix + upperTriangularMatrix';

% ȷ���Խ����ϵ�Ԫ�ز�����1
symmetricMatrix(logical(symmetricMatrix > 1)) = 1;
symmetricMatrix=symmetricMatrix.*adjMatrix;
disp(symmetricMatrix);
end
end







