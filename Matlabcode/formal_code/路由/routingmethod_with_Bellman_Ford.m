%原文 https://zhuanlan.zhihu.com/p/74944571;

%其他参考:
%概念：https://blog.csdn.net/qq_40772692/article/details/83041282?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-83041282-blog-89787628.235^v43^pc_blog_bottom_relevance_base7&spm=1001.2101.3001.4242.1&utm_relevant_index=3
%算法SPFA：https://blog.csdn.net/muxidreamtohit/article/details/7894298
%SPFA代码实现：https://blog.csdn.net/qq_35644234/article/details/61614581
%只解决最大流问题：Ford-Fulkerson算法 https://blog.csdn.net/qq_43285351/article/details/90413583
%最小费用流理论分析与LINGO计算：https://blog.csdn.net/qq_29831163/article/details/89787628
%最小代价流与最小费用流深度学习含代码与github，平台为图像识别多目标追踪：https://zhuanlan.zhihu.com/p/111397247
%网络最大流算法的python实现：https://blog.csdn.net/weixin_51545953/article/details/129009589
%相反思路：保消耗提升流 https://blog.csdn.net/qq_52852138/article/details/124633494?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-124633494-blog-89787628.235^v43^pc_blog_bottom_relevance_base7&spm=1001.2101.3001.4242.1&utm_relevant_index=3
%% 初始化
clc;
clear;
% 定义邻接矩阵
V = [
    0     5     7     0     0;
    0     0     0     3     4;
    0     2     0    10     0;
    0     0     0     0     8;
    0     0     0     0     0
];

C =[
     0     8     7     0     0
     0     0     0     2     9
     0     5     0     9     0
     0     0     0     0     4
     0     0     0     0     0
     ]; %损失矩阵
%% 创建适合matlab_maxflow()函数计算环境
% 获取矩阵的大小
[numNodes, ~] = size(V);
% 初始化边列表和权重列表
s = [];
t = [];
weights = [];
% 遍历邻接矩阵以提取边和权重
for i = 1:numNodes
    for j = 1:numNodes
        if V(i, j) ~= 0 % 如果存在边
            s = [s; i]; % 起点
            t = [t; j]; % 终点
            weights = [weights; V(i, j)]; % 权重
        end
    end
end
% 创建有向图对象
G = digraph(s, t, weights);
% 绘制有向图，并显示边的权重
plot(G, 'EdgeLabel', G.Edges.Weight, 'Layout', 'layered');
% 计算从节点 1 到节点 5 的最大流
mf = maxflow(G, 1, 5);
disp('Matlab最大流计算结果：');
disp(mf);


n=size(V,2);

%% 算法数据初始化
wf=0;wf0=Inf; %wf:最大流量, wf0:预设最大流量，这里初始化为无限大表示不限制
f = zeros(n,n); %初始化流量矩阵
while 1
%% 加权网络
    a = inf*(ones(n,n)-eye(n,n)); %有向加权图
    for i=1:n
        for j=1:n
            if V(i,j)>0&&f(i,j)==0
                a(i,j)=C(i,j);
            elseif V(i,j)>0&&f(i,j)==V(i,j)
                a(j,i)=-C(i,j);
            elseif V(i,j)>0
                a(i,j)=C(i,j);a(j,i)=-C(i,j);
            end
        end
    end
    
    %% 查找最短路径使用Bellman-Ford算法
    p = inf * ones(1,n); %初始化距离向量
    p(1) = 0;
    s = 1:n;
    s(1) = 0;
    for k=1:n
        pd=1; %标记是否更新了最短路径
        for i=2:n
            for j=1:n
                if p(i)>p(j)+a(j,i)
                    p(i)=p(j)+a(j,i);
                    s(i)=j;
                    pd=0;
                end
            end
        end
        if pd
            break;
        end
    end
    if p(n)==Inf
        disp('计算完成');
        break;
    end %如果没有找到从起点到终点的可行路径，结束循环

    %% 流量调整
    k=n;
    dvt=Inf;%dvt:可调整的最大流量
    t=n; 
    while 1 %计算可调整的最大流量
        if a(s(t),t)>0
            dvtt=V(s(t),t)-f(s(t),t); %正向弧
        elseif a(s(t),t)<0
            dvtt=f(t,s(t)); %反向弧
        end 
        if dvt>dvtt
            dvt=dvtt;
        end
        if s(t)==1
            break;
        end %当到达起点时结束
        t=s(t);
    end
    pd=0;
    if wf+dvt>=wf0 %如果当前总流量加上可调整的最大流量超过了预设的最大值
        dvt=wf0-wf;
        pd=1;
    end
    t=n;
    while 1 %调整流量
        if a(s(t),t)>0
            f(s(t),t)=f(s(t),t)+dvt; %正向弧
        elseif a(s(t),t)<0
            f(t,s(t))=f(t,s(t))-dvt; %反向弧
        end 
        if s(t)==1
            break;
        end %当到达起点时结束
        t=s(t);
    end
    if pd
        break;
    end %如果达到预设的最大流量，结束循环
end

%% 显示计算结果与结果验证
wf = sum(f(1,:));
zwf = sum(sum(C.*f)); %计算最小损失
disp('数据流矩阵')
disp(f); %流矩阵
disp('汇节点接收到的最大流量')
disp(wf);%最大流
disp('该次传输的最小消耗')
disp(zwf); %最小损失，为数据流矩阵对应单位数据量与代价矩阵对应项相乘。
if wf==mf
    disp('此次生成,设计算法与matlab最大流计算结果相同');
else
    disp('此次生成，设计算法与matlab最大流计算结果不同');
end

%% 算法流程解析
% 创建加权网络a，初始化为无穷大表示未探索，初始化对角线为0表示无自环
% 遍历每一对节点（i，j）：
%         如果存在这个边且流量为0，则设置该边权重为c（i，j）
%         如果存在边且当前流量等于容量，则设置反向边权重为-c（i，j）
%         否则同时设置正向边与反向边权重
% 使用Bellman—Ford算法，适用于存在负边的最短路径搜索
%         初始化距离向量p为无穷大，起点距离为0
%         初始化前驱向量s起点的前驱标记为0
%         迭代更新距离向量与前驱向量，使起点到其他节点的最短路径都被计算
%         如果某一轮没有任何距离更新就终止循环
%         如果最终目的地的距离仍为无穷大，说明不存在这样一条可行路径，算法结束
% 流量调整
%         初始化可调整流量为无穷大，节点为终点。
%         沿着最短路径开始回溯，计算可增加的最大流量dvt
%         更新dvt为当前最小的dvtt。
%         回溯到前一个节点，直到回到起点
%       如果当前总流量加上dvt已经超过预设流量最大值wf0，则限制dvt为使得总流量达到wf0所需要的增量
%       在沿着最短路径回溯，更新流量矩阵f，增加或者减少流量
%       如果达到了预设最大，终止循环。
%       
% 重复以上步骤，直到没有可行路径为止
% 
% 所得流量矩阵为网络中节点的实际速率分配
% 最大流wf为最后一个节点接收的总流
% 最小消耗为整个网络传输过程中的成本