function [time] = timecalculate_forDynamicwithStatic(flowMatrix,data_per_round)
   flowarry = flowprocess(flowMatrix);
    nonZeroIndices = flowarry ~= 0;
     % ���û�з���Ԫ�أ��򷵻�NaN
    if isempty(nonZeroIndices)
        time = nan;
        return;
    end
    % ����������
    divisionResults = data_per_round ./ flowarry(nonZeroIndices);
    finiteResults = divisionResults(isfinite(divisionResults));
       if isempty(finiteResults)
        time = nan;
        return;
       end
    % �������ֵ
    time = max(finiteResults);
   
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
end

