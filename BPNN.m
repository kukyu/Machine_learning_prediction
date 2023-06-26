%% 清空环境变量
close all
clear
clc
% R2_all = [];
%% 数据预处理
table = readtable('./order_weather_predict_data.csv');

% dummy = dummyvar(categorical(table.year));
% table = [table array2table(dummy)];
% table(:,"year") = [];
data = table2array(table);
%% 设定变量值
train_num = round(size(data, 1) * 0.8);
% train_num = size(data,1) - 31;

% [test_row, ~] = find(data(:,end)==2023,1); % 训练集开始行
% test_num = size(data,1) - test_row + 1;
test_row = train_num + 1;
test_num = size(data,1) - train_num;

%% 训练集和测试集
% 随机产生训练集和测试集
%% 测试1
% % 训练集
% temp = 1:1:size(data,1);
temp = randperm(size(data,1));
P_train = data(temp(1:train_num),[3,5,7:8,10,22:23])';
T_train = data(temp(1:train_num),1)';
M= size(P_train,2); 
% 
P_test = data(temp(test_row:end),[3,5,7:8,10,22:23])';
T_test = data(temp(test_row:end),1)';
N = size(P_test,2); %返回矩阵的列数
%% 数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);
%% BP 神经网络创建、训练及仿真测试
%% 创建网络
%%调用形式
RMSE_all=[];            %运行误差记录
R2_all=[];            %运行误差记录
TIME=[];                %运行时间记录
num_iter_all=10;   %随机运行次数
for NN=1:num_iter_all
t1=clock;
net = newff(p_train,t_train, 9, {'tansig','purelin'});
% 设置训练参数
net.trainParam.epochs = 100; %最大训练次数
net.trainParam.goal = 1e-3; % 训练误差最小值
net.trainParam.lr = 0.01;  % 学习速率为0.01
% 训练网络
net = train(net,p_train,t_train);
%% 仿真测试
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test);
%% 数据反归一化
T_sim1 = mapminmax('reverse', t_sim1,ps_output);
T_sim2 = mapminmax('reverse', t_sim2,ps_output);
%% 性能评价
% 均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

R2_bp = (N * sum(T_sim2 .* T_test) - sum(T_sim2) * sum(T_test))^2 / ...
((N * sum((T_sim2).^2) - (sum(T_sim2))^2) * (N * sum((T_test).^2) - ...
(sum(T_test))^2));

t2=clock;         
Time_all=etime(t2,t1);
RMSE_all=[RMSE_all,error2];
R2_all = [R2_all,R2_bp];
TIME=[TIME,Time_all];
end
%% 绘图
figure(1)
plot(RMSE_all, '-ok','LineWidth', 1.2, "MarkerEdgeColor","red", ...
    "MarkerFaceColor", "b", "MarkerSize", 3)
% 绘制折线图

xlabel('实验次数')
ylabel('均方误差')
title("每次实验的均方误差图")


figure(2)
plot(1:N,T_test,'b:*',1:N,T_sim2,'r-o')
legend('真实值','BP预测值')
xlabel('预测样本')
ylabel('出行次数')
string = {'日租赁次数预测对比图';['R^2=' num2str(R2_bp)]};
title(string)