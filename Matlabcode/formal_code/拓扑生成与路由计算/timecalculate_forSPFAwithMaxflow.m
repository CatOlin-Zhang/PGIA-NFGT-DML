function [time] = timecalculate_forSPFAwithMaxflow(flowMatrix,data_per_round)
%用于计算路由汇聚时间，按最大时间计
    numColumns = size(flowMatrix, 2);
%     data_per_round = repmat(data_per_round, 1, numColumns);
    % 初始化存储结果的数组
    transmissionTimes = zeros(1, numColumns);
    
    % 计算每个数据流的传输时间
    for i = 1:numColumns
        if flowMatrix(1, i) ~= 0
            transmissionTimes(i) = data_per_round / flowMatrix(1, i);
        else
            transmissionTimes(i) = NaN; % 不传就跳过
        end
    end
    % 返回非NaN的最大值
    time = nanmax(transmissionTimes);
end

