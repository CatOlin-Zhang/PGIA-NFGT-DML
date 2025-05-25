function W = DeltaBatch(W, X, D, learning_rate)
    % X ����������ÿһ����һ������
    % D �Ǳ�ǩ����
    % learning_rate ��ѧϰ��
    % W ��Ȩ������
    % ��ʼ���ݶ�Ϊ������
    gradient = zeros(size(W));
    
    % �����������ݼ������ݶ�
    for i = 1:size(X, 1)
        x = X(i, :)'; % ȡ��һ��������������������ת��Ϊ������
        d = D(i);     % ȡ����Ӧ�ı�ǩ
        y = Sigmoid(W * x); % ����Ԥ��ֵ
        error = d - y;      % �������
        % �ۻ��ݶ�
        gradient = gradient + error * x;
    end
    
    % ����ƽ���ݶȲ�����Ȩ��
    average_gradient = gradient / size(X, 1);
    W = W + learning_rate * average_gradient';
end