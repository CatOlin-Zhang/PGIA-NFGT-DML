clc;
clear;
%超参数
runtimes=100;

% 初始化变量

spfa_times = zeros(1,runtimes );
spfa_cost=zeros(1, runtimes);
spfa_var=zeros(1,runtimes);
spfa_pointinflowratio=zeros(1,runtimes);
spfa_maxflow=zeros(1,runtimes);

greedy_times = zeros(1, runtimes);
greedy_cost=zeros(1, runtimes);
greedy_var=zeros(1,runtimes);
greedy_pointinflowratio=zeros(1,runtimes);
greedy_maxflow=zeros(1,runtimes);

dynamic_times = zeros(1, runtimes);
dynamic_cost=zeros(1, runtimes);
dynamic_var=zeros(1,runtimes);
dynamic_pointinflowratio=zeros(1,runtimes);
dynamic_maxflow=zeros(1,runtimes);

static_times = zeros(1, runtimes);
static_cost=zeros(1, runtimes);
static_var=zeros(1,runtimes);
static_pointinflowratio=zeros(1,runtimes);
static_maxflow=zeros(1,runtimes);

parfor i = 1:runtimes
[......
 spfa_times(i),greedy_times(i),dynamic_times(i),static_times(i),......
 spfa_cost(i),greedy_cost(i),dynamic_cost(i),static_cost(i),......
 spfa_var(i),greedy_var(i),dynamic_var(i),static_var(i),......
 spfa_pointinflowratio(i),greedy_pointinflowratio(i),dynamic_pointinflowratio(i),static_pointinflowratio(i),......
 spfa_maxflow(i),greedy_maxflow(i),dynamic_maxflow(i),static_maxflow(i)......
 ]=maincode();
end
static_var(isinf(static_var)) = NaN;

% 计算平均值
average_spfa_time = mean(spfa_times);
average_spfa_cost = mean(spfa_cost);
average_spfa_var  = mean(spfa_var);
average_spfa_pointinflowratio = mean(spfa_pointinflowratio);
average_spfa_maxflow = mean(spfa_maxflow);

average_greedy_time = mean(greedy_times);
average_greedy_cost = mean(greedy_cost);
average_greedy_var  = mean(greedy_var);
average_greedy_pointinflowratio = mean(greedy_pointinflowratio);
average_greedy_maxflow = mean(greedy_maxflow);

average_dynamic_time = mean(dynamic_times);
average_dynamic_cost = mean(dynamic_cost);
average_dynamic_var  = mean(dynamic_var);
average_dynamic_pointinflowratio = mean(dynamic_pointinflowratio);
average_dynamic_maxflow = mean(dynamic_maxflow);

average_static_time = nanmean(static_times);
average_static_cost = mean(static_cost);
average_static_var  = nanmean(static_var);
average_static_pointinflowratio = mean(static_pointinflowratio);
average_static_maxflow = mean(static_maxflow);

% 输出结果
fprintf('=======================================================\n');
fprintf('SPFA路由传输时间平均值: %.6f\n', average_spfa_time);
fprintf('最大流路由传输时间平均值: %.6f\n', average_greedy_time);
fprintf('动态最短路径路由传输时间平均值: %.6f\n', average_dynamic_time);
fprintf('静态最短路径路由传输时间平均值: %.6f\n', average_static_time);
fprintf('=======================================================\n');
fprintf('SPFA路由消耗平均值: %.6f\n', average_spfa_cost);
fprintf('最大流路由消耗平均值: %.6f\n', average_greedy_cost);
fprintf('动态最短路径路由消耗平均值: %.6f\n', average_dynamic_cost);
fprintf('静态最短路径路由消耗平均值: %.6f\n', average_static_cost);
fprintf('=======================================================\n');
fprintf('SPFA路由数据流方差平均值: %.6f\n', average_spfa_var);
fprintf('最大流数据流方差平均值: %.6f\n', average_greedy_var );
fprintf('动态最短路径路由数据流方差平均值: %.6f\n', average_dynamic_var);
fprintf('静态最短路径路由数据流方差平均值: %.6f\n', average_static_var);
fprintf('=======================================================\n');
fprintf('SPFA路由汇聚比例平均值: %.6f\n', average_spfa_pointinflowratio);
fprintf('最大流汇聚比例平均值: %.6f\n', average_greedy_pointinflowratio);
fprintf('动态最短路径路由汇聚比例平均值: %.6f\n', average_dynamic_pointinflowratio);
fprintf('静态最短路径路由汇聚比例平均值: %.6f\n', average_static_pointinflowratio);
fprintf('=======================================================\n');
fprintf('SPFA路由最大流平均值: %.6f\n', average_spfa_maxflow);
fprintf('最大流路由最大流平均值: %.6f\n', average_greedy_maxflow);
fprintf('动态最短路径路由最大流平均值: %.6f\n', average_dynamic_maxflow);
fprintf('静态最短路径路由最大流平均值: %.6f\n', average_static_maxflow);


