% Effect of introducing stochasticity in NN convergence and helping with
% neuron saturation

[x, t] = iris_dataset;
%[x,t] = simplefit_dataset;
%[x,t] = cancer_dataset;
%[x,t] = wine_dataset;
%x = -1:0.5:7.5;
%t = sin(x);

%Regression Datasets
%[x,t] = house_dataset;
%[x,t] = building_dataset;
%[x,t] = engine_dataset;
% [x,t] = bodyfat_dataset;
%[x,t] = chemical_dataset;
%[x,t] = glass_dataset;

%scale balance dataset
% t = load('scale_balanceTarget.mat');
% x = load('scale_balancePattern.mat');
% t=t.scale_balance;
% x= x.x;
% 
% %Ionosphere data
% % load('ionospherePattern.mat');
% % load('ionosphereTarget.mat');
% % % 
% t=t';
% x=x';

%prima indian diabetics
% t = load('pimaindiansdiabetesTarget.mat');
% t = t.pimaindiansdiabetesTarget;
% x = load('primaindianInput.mat');
% x = x.primaindianInput;
% x=x';
% t=t';


%HeartC data
% load('heartPattern.mat')
% load('heartTarget.mat')
% x=x';
% t=t';
%[x,t] =thyroid_dataset;

[I N] = size(x);
[O N]=size(t);
maxt = max(max(t));  

Q = size(x,2); %total number of samples
Q1 = floor(Q * 0.80); %90% for training
Q2 = Q-Q1; %10% for testing
ind = randperm(Q);
ind1 = ind(1:Q1);
ind2 = ind(Q1 + (1:Q2));
x1 = x(:, ind1);
t1 = t(:, ind1);
x2 = x(:, ind2);
t2 = t(:, ind2);
count = 1000;
goal = 0.0;
momentum = 0.01;
learning_rate = 0.1;
noise = 1.0;
mingrad = 1e-07;
numNN = 1; %repetition


% nettansig.layers{1}.transferFcn = 'tansig';
% nettansig.layers{2}.transferFcn = 'tansig';
% netelliot.layers{1}.transferFcn = 'elliotsig';
% netelliot.layers{2}.transferFcn = 'elliotsig';
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
%trainFcn = 'trainrp';  % Resilient backpropagation. 
%trainFcn = 'traingd';  % traditional backpropagation. 
%trainFcn = 'traingdm';  %  backpropagation with momentum. 

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
H = hiddenLayerSize;

for i = 1:numNN %to create 100 networks with 100 initialization
net = feedforwardnet(hiddenLayerSize,trainFcn);
net = configure(net,x1,t1);

%Elliotsig Network

netelliot = net;
netelliot.layers.transferFcn={'elliotsig';'purelin'};
netelliot.input.processFcns = {}; %{'removeconstantrows');
netelliot.output.processFcns = {};
netelliot.divideFcn = '';
netelliot.trainParam.goal = goal;
netelliot.trainParam.epochs = count;
netelliot.trainParam.min_grad = mingrad;
netelliot.trainParam.mc = momentum;
netelliot.trainParam.lr = learning_rate;


%Elliotsig Network with stochasticity
netelliotR = net;
netelliotR.layers.transferFcn={'elliotsigR';'purelin'};
netelliotR.input.processFcns = {}; %{'removeconstantrows');
netelliotR.output.processFcns = {};
netelliotR.divideFcn = '';
netelliotR.trainParam.goal = goal;
netelliotR.trainParam.epochs = count;
netelliotR.trainParam.min_grad = mingrad;
netelliotR.trainParam.mc = momentum;
netelliotR.trainParam.lr = learning_rate;

%Tansig Network

nettansig = net;
nettansig.layers.transferFcn={'tansig';'purelin'};
nettansig.input.processFcns = {}; %{'removeconstantrows');
nettansig.output.processFcns = {};
nettansig.divideFcn = '';
nettansig.trainParam.goal = goal;
nettansig.trainParam.epochs = count;
nettansig.trainParam.min_grad = mingrad;
nettansig.trainParam.mc = momentum;
nettansig.trainParam.lr = learning_rate;

%Tansig Network with stochasticity

nettansigR = net;
nettansigR.layers.transferFcn={'tansigR';'purelin'};
nettansigR.input.processFcns = {}; %{'removeconstantrows');
nettansigR.output.processFcns = {};
nettansigR.divideFcn = '';
nettansigR.trainParam.goal = goal;
nettansigR.trainParam.epochs = count;
nettansigR.trainParam.min_grad = mingrad;
nettansigR.trainParam.mc = momentum;
nettansigR.trainParam.lr = learning_rate;
%nettansigR.activation_noise = 0.1;
%nettansigR.dd_offset = 0;

NNelliot = cell(1, numNN);
NNelliotR = cell(1, numNN);
NNtansig = cell(1, numNN);
NNtansigR = cell(1, numNN);
perfs = zeros(1, numNN);

%WEight manual Initialization at the bottom of the script


    [NNelliot{i}, trelliot{i}] = train(netelliot, x1, t1);
    [NNelliotR{i}, trelliotR{i}] = train(netelliotR, x1, t1);
    [NNtansig{i}, trtansig{i}] = train(nettansig, x1, t1);
    [NNtansigR{i}, trtansigR{i}] = train(nettansigR, x1, t1);
    %y2 = NN{i}(x2); % same as y2 = sim(NN{i},x2);
    y1elliot = NNelliot{i}(x1);
    y1elliotR = NNelliotR{i}(x1);
    y1tansig = NNtansig{i}(x1);
    y1tansigR = NNtansigR{i}(x1);
    
    perfselliottrain(i) = mse(netelliot, t1, y1elliot);
    perfselliotRtrain(i) = mse(netelliotR, t1, y1elliotR);
    perfstansigtrain(i) = mse(nettansig, t1, y1tansig);
    perfstansigRtrain(i) = mse(nettansigR, t1, y1tansigR);
    
    y2elliot = NNelliot{i}(x2);
    y2elliotR = NNelliotR{i}(x2);
    y2tansig = NNtansig{i}(x2);
    y2tansigR = NNtansigR{i}(x2);
    
    perfselliottest(i) = mse(netelliot, t2, y2elliot);
    perfselliotRtest(i) = mse(netelliotR, t2, y2elliotR);
    perfstansigtest(i) = mse(nettansig, t2, y2tansig);
    perfstansigRtest(i) = mse(nettansigR, t2, y2tansigR);
    
     epochelliot1(i) = trelliot{1,i}.num_epochs;
     epochelliotR1(i) = trelliotR{1,i}.num_epochs;
     epochtansig1(i) = trtansig{1,i}.num_epochs;
     epochtansigR1(i) = trtansigR{1,i}.num_epochs;
    
     epochelliot = trelliot{1,i}.num_epochs;
     epochelliotR = trelliotR{1,i}.num_epochs;
     epochtansig = trtansig{1,i}.num_epochs;
     epochtansigR = trtansigR{1,i}.num_epochs;
       
     minperfselliottrain = find(perfselliottrain==min(perfselliottrain));
     minperfselliotRtrain = find(perfselliotRtrain==min(perfselliotRtrain));
     minperfstansigtrain = find(perfstansigtrain==min(perfstansigtrain));
     minperfstansigRtrain = find(perfstansigRtrain==min(perfstansigRtrain));
     
     minperfselliottest = find(perfselliottest==min(perfselliottest));
     minperfselliotRtest = find(perfselliotRtest==min(perfselliotRtest));
     minperfstansigtest = find(perfstansigtest==min(perfstansigtest));
     minperfstansigRtest = find(perfstansigRtest==min(perfstansigRtest));
     
     fprintf('Training %d/%d, %d, %d, %d, %d\n', i , numNN, epochelliot, epochelliotR, epochtansig ,epochtansigR);
end



minperfselliottest
minperfselliotRtest
minperfstansigtest
minperfstansigRtest

perfselliottest
perfselliotRtest
perfstansigtest
perfstansigRtest

meanpefelliottest = mean(perfselliottest)
meanpefelliotRtest = mean(perfselliotRtest)
meanpeftansigtest = mean(perfstansigtest)
meanpeftansigRtest = mean(perfstansigRtest)


mean(epochelliot1)
mean(epochelliotR1)
mean(epochtansig1)
mean(epochtansigR1)



%     rng(i)
% 
% netelliot = configure(netelliot, x, t );
% %rng(161)
% 
% IW = noise*randn(H,I);
% 
% b1 = noise*randn(H,1);
% 
% LW = noise*randn(O,H);
% 
% b2 = noise*randn(O,1);
% 
% netelliot.IW{1,1} = IW;
% 
% netelliot.b{1,1} = b1;
% 
% netelliot.LW{2,1} = LW;
% 
% netelliot.b{2,1} = b2;
% 
% 
% 

% nettansig = configure(nettansig, x, t );
% %rng(161)
% 
% IW = noise*randn(H,I);
% 
% b1 = noise*randn(H,1);
% 
% LW = noise*randn(O,H);
% 
% b2 =noise*randn(O,1);
% 
% nettansig.IW{1,1} = IW;
% 
% nettansig.b{1,1} = b1;
% 
% nettansig.LW{2,1} = LW;
% 
% nettansig.b{2,1} = b2;
