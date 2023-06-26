%% 清空环境变量
close all
clear
clc
%% 数据预处理
table = readtable('../order_weather_predict_data.csv');

% dummy = dummyvar(categorical(table.year));
% table = [table array2table(dummy)];
% table(:,"year") = [];
data = table2array(table);
%% 设定变量值
train_num = round(size(data, 1) * 0.8);
% train_num = size(data,1) - 31;

test_row = train_num + 1;
% [test_row, ~] = find(data(:,end)==2023,1); % 训练集开始行
% test_num = size(data,1) - test_row + 1;
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
num_iter_all=1;   %随机运行次数
%% 循环
for NN=1:num_iter_all
input_num = size(p_train, 1);     %输入特征个数
hidden_num = 9;   %隐藏层神经元个数
output_num = size(t_train, 1); %输出特征个数
t1=clock;
%% 遗传算法初始化
iter_num=100;                         %总体进化迭代次数
group_num=10;                      %种群规模
cross_pro=0.4;                       %交叉概率
mutation_pro=0.05;                  %变异概率，相对来说比较小
num_all=input_num*hidden_num+hidden_num+hidden_num*output_num+output_num;%网络总参数，只含一层隐藏层
lenchrom=ones(1,num_all);  %种群总长度
limit=[-2*ones(num_all,1) 2*ones(num_all,1)];    %初始参数给定范围
%% 初始化种群
input_data = p_train;
output_data = t_train;
for i=1:group_num
    initial=rand(1,length(lenchrom));  %产生0-1的随机数
    initial_chrom(i,:)=limit(:,1)'+(limit(:,2)-limit(:,1))'.*initial; %变成染色体的形式，一行为一条染色体
    fitness_value=fitness(initial_chrom(i,:),input_num,hidden_num,output_num,input_data,output_data);
    fitness_group(i)=fitness_value;
end
[bestfitness,bestindex]=min(fitness_group);
bestchrom=initial_chrom(bestindex,:);  %最好的染色体
avgfitness=sum(fitness_group)/group_num; %染色体的平均适应度                              
trace=[avgfitness bestfitness]; % 记录每一代进化中最好的适应度和平均适应度
%% 迭代过程
new_chrom=initial_chrom;
new_fitness=fitness_group;
 for num=1:iter_num
    % 选择  
     [new_chrom,new_fitness]=select(new_chrom,new_fitness,group_num);   %把表现好的挑出来，还是和种群数量一样
    %交叉  
     new_chrom=Cross(cross_pro,lenchrom,new_chrom,group_num,limit);
    % 变异  
     new_chrom=Mutation(mutation_pro,lenchrom,new_chrom,group_num,num,iter_num,limit);     
    % 计算适应度   
    for j=1:group_num  
        sgroup=new_chrom(j,:); %个体 
        new_fitness(j)=fitness(sgroup,input_num,hidden_num,output_num,input_data,output_data);     
    end  
    %找到最小和最大适应度的染色体及它们在种群中的位置
    [newbestfitness,newbestindex]=min(new_fitness);
    [worestfitness,worestindex]=max(new_fitness);
    % 代替上一次进化中最好的染色体
    if  bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=new_chrom(newbestindex,:);
    end
    new_chrom(worestindex,:)=bestchrom;
    new_fitness(worestindex)=bestfitness;
    avgfitness=sum(new_fitness)/group_num;
    trace=[trace;avgfitness bestfitness]; %记录每一代进化中最好的适应度和平均适应度
 end
%%
figure(1)
[r ,~]=size(trace);
plot([1:r]',trace(:,2),'b--');
title(['适应度曲线  ' '终止代数＝' num2str(iter_num)]);
xlabel('进化代数');ylabel('适应度');
legend('最佳适应度');
 
%% 把最优初始阀值权值赋予网络预测
% %用遗传算法优化的BP网络进行值预测
net=newff(p_train,t_train,hidden_num,{'tansig','purelin'},'trainlm');
w1=bestchrom(1:input_num*hidden_num);   %输入和隐藏层之间的权重参数
B1=bestchrom(input_num*hidden_num+1:input_num*hidden_num+hidden_num); %隐藏层神经元的偏置
w2=bestchrom(input_num*hidden_num+hidden_num+1:input_num*hidden_num+...
    hidden_num+hidden_num*output_num);  %隐藏层和输出层之间的偏置
B2=bestchrom(input_num*hidden_num+hidden_num+hidden_num*output_num+1:input_num*hidden_num+...
    hidden_num+hidden_num*output_num+output_num); %输出层神经元的偏置
%网络权值赋值
net.iw{1,1}=reshape(w1,hidden_num,input_num);
net.lw{2,1}=reshape(w2,output_num,hidden_num);
net.b{1}=reshape(B1,hidden_num,1);
net.b{2}=reshape(B2,output_num,1);
%% 神经网络参数设置
net.trainParam.epochs = 200;          % 最大迭代次数
net.trainParam.lr = 0.01;              % 学习率
net.trainParam.goal=0.001;
[net,~]=train(net,p_train,t_train);
%% 进行预测
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test);
%% 数据反归一化
T_sim1 = mapminmax('reverse', t_sim1,ps_output);
T_sim2 = mapminmax('reverse', t_sim2,ps_output);
%% 性能评价
% 均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

R2 = (N * sum(T_sim2 .* T_test) - sum(T_sim2) * sum(T_test))^2 / ...
((N * sum((T_sim2).^2) - (sum(T_sim2))^2) * (N * sum((T_test).^2) - ...
(sum(T_test))^2));

t2=clock;         
Time_all=etime(t2,t1);
RMSE_all=[RMSE_all,error2];
R2_all = [R2_all,R2];
TIME=[TIME,Time_all];
end
%%
% figure(1)
% plot(RMSE_all,'LineWidth',2)
% xlabel('实验次数')
% ylabel('误差')

figure(2)
plot(1:N,T_test,'b:*',1:N,T_sim2,'r-o')
legend('真实值','预测值')
xlabel('预测样本')
ylabel('出行次数')
string = {'日租赁次数预测对比图';['R^2=' num2str(R2)]};
title(string)

        


