function W = DeltaSGD(W, X, D, learning_rate)
    % ���� X �ǵ�������������������D �Ƕ�Ӧ�ı�ǩ
    % ����ʵ�ֵ��ǻ��ڵ�������������ݶ��½�����
    x = X(:); % ����������ת��Ϊ������
    d = D;    % ��ǩ
    y = Sigmoid(W * x); % Ԥ��ֵ
    error = d - y;      % ���
    gradient = error * x; % �ݶȣ��Ե���������
    W = W + learning_rate * gradient'; % ����Ȩ��
end
