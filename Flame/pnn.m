clc;
clear;
close all;

%% get data
data = importdata('Flame.txt');

%% shuffle data
[m, n] = size(data);
[~, idx] = sort(rand(m,1));

for c = 1:m
    data(c,1:3) = data(idx(c),1:3);
end

%% splite Data 70/30
PD = 0.70;
trainn = data(1:round(PD*length(data)),:); 
test = data(round(PD*length(data)):end,:);

%% train
colmn1_train=trainn(:,1);
colmn2_train=trainn(:,2);

colmn1_norm_train=(colmn1_train - mean(colmn1_train)) ./ std(colmn1_train);
colmn2_norm_train=(colmn2_train - mean(colmn2_train)) ./ std(colmn2_train);

x_train = cat(2,colmn1_norm_train,colmn2_norm_train);

x_train = transpose(x_train);
y_train = transpose(trainn(:,3));


%% build model and train

Spread=0.2;
y_train=ind2vec(y_train);
net = newpnn(x_train,y_train,Spread);


net.divideFcn = 'dividerand';  
net.divideMode = 'sample'; 

% splite train data for val and train data
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;

net.trainFcn = 'traingd'; 
net.performFcn = 'mse'; 

net.plotFcns = {'plotregression','plotfit'};

net.trainParam.showWindow=true;
net.trainParam.showCommandLine=false;
net.trainParam.show=1;
net.trainParam.lr=0.003;
net.trainParam.epochs=30;
net.trainParam.goal=1e-8;
net.trainParam.max_fail=20;

%% train the network
[net,tr] = train(net,x_train,y_train);


%% test normalaze and splite 
colmn1_test = test(:,1);
colmn2_test = test(:,2);

colmn1_norm_test = (colmn1_test - mean(colmn1_test)) ./ std(colmn1_test);
colmn2_norm_test = (colmn2_test - mean(colmn2_test)) ./ std(colmn2_test);

x_test = cat(2,colmn1_norm_test,colmn2_norm_test);

x_test = transpose(x_test);
y_test = transpose(test(:,3));

%% test 
Y = net(x_test);
Y=vec2ind(Y);
%% evaluate
pred = round(Y);
acc_count = nnz(pred==y_test); 
acc = acc_count/length(y_test);

disp("accuracu : " + acc);