function netClassify

% Load data sample
load('C:\Data\2017_07_MI_errP\S03\20170721T143812.mat');

% Create target matrix
target=zeros(length(obj.rawData.data),obj.nClasses+1);
for currStim=1:length(obj.MItimeStamps)
    target(obj.MItimeStamps(currStim)*obj.fs,obj.trialLbls(currStim)+1)=1;
end
B=ones(obj.fs*obj.timingParams.cue,1); % Rectangular window of obj.timingParams.cue seconds
for currTarget=2:obj.nClasses+1
    target(:,currTarget)=filter(B,1,target(:,currTarget));
end
B=blackman(obj.fs/4); % Prevent sharp changes: filter with blackman window of .25s
B=B/sum(B);
for currTarget=2:obj.nClasses+1
    target(:,currTarget)=filter(B,1,target(:,currTarget));
end
target(:,1)=1-sum(target,2);

% Hard preproc
freqData=MI_session.freqFilter(obj.rawData.data,obj.fs,[1,30]);
nClasses=4;
R=cov(freqData);
Rclass=cell(1,nClasses);
for currClass=1:nClasses
    Rclass{currClass}=cov(freqData(target(:,currClass+1)>.99,:));
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

% Selct which features to use
nFeats=2*obj.nClasses; % This is the number of features to use. It is set to 2*nClasses in a totally arbitrary way
I=zeros(1,length(V));
[~,targetVec]=max(target,[],2);
targetVec(targetVec==1)=[];
Pclass=histcounts(targetVec,'Normalization','pdf');
for currCol=1:length(V)
    currV=V(:,currCol)/sqrt(V(:,currCol)'*R*V(:,currCol));
    vRv=cellfun(@(x)(currV'*x*currV),Rclass);
    I(currCol)=-sum(Pclass.*log(sqrt(vRv)))-3/16*sum(Pclass.*(vRv.^2-1))^2;
end
[~,relIdxs]=sort(I,'descend');
relIdxs=relIdxs(1:nFeats); 
Vrel=V(:,relIdxs);

% Split data
trainStart=3e4+1;
trainEnd=trainStart+2e5-1;
rawTrainData=obj.rawData.data(trainStart:trainEnd,:);
preprocTrainData=freqData(trainStart:trainEnd,:)*Vrel;
targetTrainData=target(trainStart:trainEnd,:);

% Train first network (spatial and temporal filtering)
preprocNet=trainNet(rawTrainData,preprocTrainData,'regression'); % Last parameters is needed to suppress output normalization

% Apply fist network on input data
preprocTrainDataEst=cell2mat(preprocNet(tonndata(rawTrainData,false,false)))';

% Train second network to get target
targetNet=trainNet(preprocTrainDataEst,targetTrainData,'classification'); % Last parameters is needed to suppress input normalization

% Get target estimation from second net
targetTrainEst=targetNet(tonndata(preprocTrainDataEst,false,false));

% Stack two networks together
oneNet=timedelaynet(1:8,[preprocNet.layers{1}.size,preprocNet.layers{2}.size,targetNet.layers{1}.size]);
oneNet=configure(oneNet,rawTrainData',targetTrainData');
oneNet.layerWeights{3,2}.delays=0:7;
oneNet.layers{2}.transferFcn='purelin';
oneNet.layers{end}.transferFcn='logsig';
oneNet.IW{1}=preprocNet.IW{1};
oneNet.LW{2,1}=preprocNet.LW{2,1};
oneNet.LW{3,2}=targetNet.IW{1};
oneNet.LW{4,3}=targetNet.LW{2,1};
oneNet.b{1}=preprocNet.b{1};
oneNet.b{2}=preprocNet.b{2};
oneNet.b{3}=targetNet.b{1};
oneNet.b{4}=targetNet.b{2};

% Fine-tune stacked network
oneNetTrained=train(oneNet,tonndata(rawTrainData,false,false),tonndata(targetTrainData,false,false),'trainrp');

% Results from fine-tuned net
targetTrainEstOneNet=oneNetTrained(tonndata(rawTrainData,false,false));

% Plot results
hold on
for currClass=1:5
    plot(targetTrainData(:,currClass)*.9+currClass-1,'k')
    plot(targetTrainEst(:,currClass)*.9+currClass-1,'r')
    plot(targetTrainEstOneNet(:,currClass)*.9+currClass-1,'g')
end

keyboard;
end

function net=trainNet(trainData,trainTarget,taskType)
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
inputDelays = 1:8;
hiddenLayerSize = 10;
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
        net.layers{end}.transferFcn='logsig';
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
net.divideFcn = 'divideblock';  % Divide data randomly
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