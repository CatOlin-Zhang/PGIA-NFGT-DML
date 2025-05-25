function [time] = timecalculate_forSPFAwithMaxflow(flowMatrix,data_per_round)
%���ڼ���·�ɻ��ʱ�䣬�����ʱ���
    numColumns = size(flowMatrix, 2);
%     data_per_round = repmat(data_per_round, 1, numColumns);
    % ��ʼ���洢���������
    transmissionTimes = zeros(1, numColumns);
    
    % ����ÿ���������Ĵ���ʱ��
    for i = 1:numColumns
        if flowMatrix(1, i) ~= 0
            transmissionTimes(i) = data_per_round / flowMatrix(1, i);
        else
            transmissionTimes(i) = NaN; % ����������
        end
    end
    % ���ط�NaN�����ֵ
    time = nanmax(transmissionTimes);
end

