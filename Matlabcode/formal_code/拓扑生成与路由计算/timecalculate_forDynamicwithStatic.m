function [time] = timecalculate_forDynamicwithStatic(flowMatrix,data_per_round)
   flowarry = flowprocess(flowMatrix);
    nonZeroIndices = flowarry ~= 0;
     % 如果没有非零元素，则返回NaN
    if isempty(nonZeroIndices)
        time = nan;
        return;
    end
    % 计算除法结果
    divisionResults = data_per_round ./ flowarry(nonZeroIndices);
    finiteResults = divisionResults(isfinite(divisionResults));
       if isempty(finiteResults)
        time = nan;
        return;
       end
    % 返回最大值
    time = max(finiteResults);
   
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
end

