function [flowMatrix,wf,point_in_flow_ratio,variance,total_costsum] = mincost_static(adjMatrix,cost_mat,data_per_round)
 n = size(adjMatrix, 1);   
 flowMatrix = mincost_static(adjMatrix, data_per_round) ;

%  disp(flowMatrix);
 
 
%���������
wf=sum(flowMatrix(1,:));
flowarray = flowprocess(flowMatrix);
% disp(flowMatrix);
%��۱�������
pointinflow = sum(flowarray(:) > 0);
 fprintf('��̬���·����۱���%d / %d\n',pointinflow,n);
point_in_flow_ratio = pointinflow / n;
 
%���ļ���
   total_costMat = flowMatrix.*cost_mat;
   total_costsum = nansum(total_costMat(:));
   fprintf("��̬���·���������ģ�%f\n",total_costsum );


%�������

    nonZeroElements = flowarray(flowarray>=0);
    variance = var(nonZeroElements) / pointinflow;
    fprintf("��̬���·�������󷽲%f\n",variance);
   
    
    
   function flowarry = flowprocess(FlowMat)
            % ����ÿ�е��к�
    rowSums = sum(FlowMat, 2);
    
    % ����ÿ�е��к�
    colSums = sum(FlowMat, 1);
    
    % ��ʼ��һ���µ��������洢���
    flowarry = zeros(size(rowSums));
    
    % �����кͼ�ȥ�кͲ�������洢��resultArray��
    for b = 1:size(FlowMat, 1)
        flowarry(b) = rowSums(b) - colSums(b);
    end
    end
   function [flowMatrix] = mincost_static(adjMatrix,data_per_round)
    % adjMatrix: ͼ���ڽӾ���
    % cost_mat: �ɱ�����
    % upper_bandwidth: ���޴������
    % data_per_round: ÿ���ڵ�ÿ�ֿ����������������������

    n = size(adjMatrix, 1); % ͼ�нڵ������
    lastNode = n; % ���һ���ڵ�
    maxBandwidths = zeros(1, n); % ��ʼ������������
    paths = cell(n, 1); % �洢·��
    flowMatrix = zeros(n, n); % ��ʼ��������

    currentAdjMatrix = adjMatrix; 

    for startNode = 1:n-1 % ����Ҫ�����һ���ڵ㿪ʼ
        bandwidth = zeros(1, n); % ��ʼ����������
        bandwidth(startNode) = inf; % ��ʼ�ڵ�Ĵ�����Ϊ�����
        visited = false(1, n); % ��¼�ڵ��Ƿ񱻷��ʹ�
        parent = zeros(1, n); % ���ڵ��������ڼ�¼·��

        u = startNode;
        while u ~= lastNode && ~all(visited)
            visited(u) = true;
            nextNode = find(currentAdjMatrix(u, :) > 0 & ~visited, 1);
            if isempty(nextNode)
                break; % û�п���·��
            end
            parent(nextNode) = u;
            u = nextNode;
        end

        % ����·��
        path = [];
        node = lastNode;
        while node ~= 0
            path = [node, path];
            node = parent(node);
        end
        paths{startNode} = path;

        % ����ڵ㵽������������
        if length(path) == 1
            maxBandwidths(startNode) = 0;
        else
            % �ҵ�·���ϵ���С����
            minPathBandwidth = inf;
            for j = 1:length(path)-1
                u = path(j);
                v = path(j+1);
                if currentAdjMatrix(u, v) < minPathBandwidth
                    minPathBandwidth = currentAdjMatrix(u, v);
                end
            end

            % Ӧ�ô������޺� data_per_round
            allocatedBandwidth = min(minPathBandwidth, data_per_round);

            % ���������ô���
            maxBandwidths(startNode) = allocatedBandwidth;

            % ����������
            for j = 1:length(path)-1
                u = path(j);
                v = path(j+1);
                flowMatrix(u, v) = flowMatrix(u, v) + allocatedBandwidth;
            end

            % �۳�·���ϵĴ���
            for j = 1:length(path)-1
                u = path(j);
                v = path(j+1);
                currentAdjMatrix(u, v) = currentAdjMatrix(u, v) - allocatedBandwidth;
                currentAdjMatrix(v, u) = currentAdjMatrix(v, u) - allocatedBandwidth;
            end
        end
    end

% ���ÿ���ڵ㵽���һ���ڵ�������ô����·��
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
end

end







