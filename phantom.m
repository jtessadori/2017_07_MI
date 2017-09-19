function errs=phantom
close all

% Parameters
netParams.nDelays=8;
netParams.trainingSamples=1e5;
netParams.hiddenLayerSize=4;

% Generate a target and trainingData sequences
[trainTarget,trainData,testTarget,testData]=generatePhantom(netParams);

% Frequency filter training data
freqData=MI_session.freqFilter(trainData,512,[10,30]);

% Compute covariance for each class
nClasses=4;
% R=cov(freqData);
Rclass=cell(1,nClasses);
for currClass=1:nClasses
    Rclass{currClass}=cov(freqData(trainTarget(:,currClass+1)>.99,:));
end

% From now on following algorithms as described on "Multi-class Common
% Spatial Patterns and Information Theoretic Feature Extraction"

% Compute joint approximate diagonalization (see function for details)
varString=[];
for currClass=1:nClasses
    varString=[varString,sprintf('Rclass{%d},',currClass)]; %#ok<AGROW>
end
varString(end)=[];
V=eval(['simdiag(',varString,');']);
V=abs(V);

% Train first network (spatial and temporal filtering, need to add
% reduction of features)
preprocNet=trainNet(trainData,freqData*V,'regression',netParams); % Last parameters is needed to suppress output normalization

% Apply first network on input data
X=tonndata(trainData,false,false);
Yest=cell2mat(preprocNet(X))';

% Train second network to get target
targetNet=trainNet(Yest,trainTarget,'classification',netParams); % Last parameters is needed to suppress input normalization

X=tonndata(Yest,false,false);
trainTargetEst=cell2mat(targetNet(X))';

clear oneNet
oneNet=timedelaynet(1:netParams.nDelays,[preprocNet.layers{1}.size,preprocNet.layers{2}.size,targetNet.layers{1}.size],'trainrp');
oneNet=configure(oneNet,trainData',trainTarget');
oneNet.layerWeights{3,2}.delays=0:netParams.nDelays-1;
% oneNet.biasConnect(end)=0;
oneNet.layers{end}.transferFcn='tansig';
oneNet.divideFcn = 'divideblock';  % Divide data in blocks
oneNet.layers{2}.transferFcn='purelin';
oneNetRaw=oneNet;
oneNet.IW{1}=preprocNet.IW{1};
oneNet.LW{2,1}=preprocNet.LW{2,1};
oneNet.LW{3,2}=targetNet.IW{1};
oneNet.LW{4,3}=targetNet.LW{2,1};
oneNet.b{1}=preprocNet.b{1};
oneNet.b{2}=preprocNet.b{2};
oneNet.b{3}=targetNet.b{1};
oneNet.b{4}=targetNet.b{2};

% Verify that two networks applied alone and single stacked network provide
% same results
%plot(trainTargetEstOneNet(:,1)),hold on,plot(trainTargetEst(:,1))

% Fine tune stacked network
[x,xi,ai,t] = preparets(oneNet,tonndata(trainData,false,false),tonndata(trainTarget,false,false));
oneNetTrained=train(oneNet,x,t,xi,ai);
trainTargetOneNetEst=cell2mat(oneNetTrained(tonndata(trainData,false,false)))';

% Try training stacked network from scratch
oneNetRaw=train(oneNetRaw,x,t,xi,ai);
trainTargetOneNetRawEst=cell2mat(oneNetRaw(tonndata(trainData,false,false)))';

% Test fine-tuned network against single ones
processedTestData=cell2mat(preprocNet(tonndata(testData,false,false)))';
testTargetEst=cell2mat(targetNet(tonndata(processedTestData,false,false)))';
testTargetOneNetEst=cell2mat(oneNetTrained(tonndata(testData,false,false)))';
testTargetOneNetRawEst=cell2mat(oneNetRaw(tonndata(testData,false,false)))';

plot(trainTarget(:,1));
hold on;
plot(trainTargetEst(:,1));
plot(trainTargetOneNetEst(:,1));
plot(trainTargetOneNetRawEst(:,1));
legend({'Target','Single net est.','Fine-tuned stacked net est.','Stacked net est.'})

% Do same thing for testing data
figure;
plot(testTarget(:,1));
hold on;
plot(testTargetEst(:,1));
plot(testTargetOneNetEst(:,1));
plot(testTargetOneNetRawEst(:,1));
legend({'Target','Single net est.','Fine-tuned stacked net est.','Stacked net est.'})

% Evaluate performance as mean of squared errors on all targets
errs(1,1)=mean(mean((trainTarget).^2));
errs(2,1)=mean(mean((trainTargetEst-trainTarget).^2));
errs(3,1)=mean(mean((trainTargetOneNetEst-trainTarget).^2));
errs(4,1)=mean(mean((trainTargetOneNetRawEst-trainTarget).^2));
errs(1,2)=mean(mean((testTarget).^2));
errs(2,2)=mean(mean((testTargetEst-testTarget).^2));
errs(3,2)=mean(mean((testTargetOneNetEst-testTarget).^2));
errs(4,2)=mean(mean((testTargetOneNetRawEst-testTarget).^2));
errs=errs./repmat(errs(1,:),size(errs,1),1);
errs=errs(2:end,:);
disp(errs);
end

function [trainTarget,trainData,testTarget,testData]=generatePhantom(netParams)
% Generate possible rhythms
seqLength=netParams.trainingSamples;
fs=512;
nRhythms=6;
Fs=rand(nRhythms,1)*20+10; % Frequencies of possible rhytms, between 10 and 30 Hz
t=linspace(0,seqLength/fs,seqLength)';
sinWaves=sin(2*pi*t*Fs');

% Rhytms to classes mixing
nClasses=4; % plus rest class
A=rand(nRhythms,nClasses+1);

% Generate transition matrix
desiredTrials=90;
trMatrix(1,:)=[1-desiredTrials/seqLength,ones(1,nClasses)*desiredTrials/(seqLength*nClasses)];
for currClass=1:nClasses
    trMatrix(currClass+1,:)=[desiredTrials/seqLength,zeros(1,currClass-1),1-desiredTrials/seqLength,zeros(1,nClasses-currClass)];
end

% Generate target sequence
target=zeros(seqLength,nClasses+1);
currState=1;
for currT=2:length(target)
    currState=sum(rand>cumsum(trMatrix(currState,:)))+1;
    target(currT,currState)=1;
end
B=blackman(fs/4)/sum(blackman(fs/4));
target=filter(B,1,target);

% Generate training data. It consists of a mix of sin waves whose power
% change depending on target sequence
data=sinWaves*A.*target;

% Spread the information on target on several different channels
nChannels=8;
data=data*rand(nClasses+1,nChannels);

% Normalize trainData
data=data-repmat(mean(data),length(data),1);
data=data./repmat(std(data),length(data),1);

% Add noise to training data
SNR=2;
data=randn(size(data))/sqrt(SNR)+data;

% Split data into training and testing sets
trainTarget=target(1:seqLength/2,:);
trainData=data(1:seqLength/2,:);
testTarget=target(seqLength/2+1:end,:);
testData=data(seqLength/2+1:end,:);
end

function net=trainNet(trainData,trainTarget,taskType,netParams)
X = tonndata(trainData,false,false);
T = tonndata(trainTarget,false,false);

% Some features of the network depend on the network to be trained.
switch taskType
    case 'regression'
        trainFcn = 'trainbfg';
    case 'classification'
        trainFcn = 'trainrp';
end

% Create a Time Delay Network
inputDelays = 1:netParams.nDelays;
hiddenLayerSize = netParams.hiddenLayerSize;
net = timedelaynet(inputDelays,hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Suppression of normalization has got nothing to do with the task type,
% but is required here to allow stacking networks later on
switch taskType
    case 'regression'
        net.output.processFcns={};
    case 'classification'
        net.input.processFcns={};
%         net.biasConnect(end)=0;
        net.layers{end}.transferFcn='tansig';
end

% Prepare the Data for Training and Simulation
% The function PREPARETS prepares timeseries data for a particular network,
% shifting time by the minimum amount to fill input states and layer
% states. Using PREPARETS allows you to keep your original time series data
% unchanged, while easily customizing it for networks with differing
% numbers of delays, with open loop or closed loop feedback modes.
[x,xi,ai,t] = preparets(net,X,T);

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'divideblock';  % Divide data in blocks
net.divideMode = 'time';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Train the Network
% net = train(net,x,t,xi,ai,'useParallel','yes','useGPU','yes','showResources','yes');
net = train(net,x,t,xi,ai);
end

% function net=trainNet(trainData,trainTarget,suppressNormalization)
% X = tonndata(trainData,false,false);
% T = tonndata(trainTarget,false,false);
% 
% % Choose a Training Function
% % For a list of all training functions type: help nntrain
% % 'trainlm' is usually fastest.
% % 'trainbr' takes longer but may be better for challenging problems.
% % 'trainscg' uses less memory. Suitable in low memory situations.
% trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
% 
% % Create a Time Delay Network
% inputDelays = 1:8;
% hiddenLayerSize = 10;
% net = timedelaynet(inputDelays,hiddenLayerSize,trainFcn);
% 
% % Choose Input and Output Pre/Post-Processing Functions
% % For a list of all processing functions type: help nnprocess
% net.input.processFcns = {'removeconstantrows','mapminmax'};
% net.output.processFcns = {'removeconstantrows','mapminmax'};
% if suppressNormalization==1
%     net.input.processFcns={};
% end
% if suppressNormalization==2
%     net.output.processFcns={};
% end
% 
% % Prepare the Data for Training and Simulation
% % The function PREPARETS prepares timeseries data for a particular network,
% % shifting time by the minimum amount to fill input states and layer
% % states. Using PREPARETS allows you to keep your original time series data
% % unchanged, while easily customizing it for networks with differing
% % numbers of delays, with open loop or closed loop feedback modes.
% [x,xi,ai,t] = preparets(net,X,T);
% 
% % Setup Division of Data for Training, Validation, Testing
% % For a list of all data division functions type: help nndivide
% net.divideFcn = 'divideblock';  % Divide data randomly
% net.divideMode = 'time';  % Divide up every sample
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
% 
% % Choose a Performance Function
% % For a list of all performance functions type: help nnperformance
% net.performFcn = 'mse';  % Mean Squared Error
% 
% % Train the Network
% net = train(net,x,t,xi,ai);
% end