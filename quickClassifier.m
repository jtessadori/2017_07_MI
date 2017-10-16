% Testing simple classifier. Load file, apply lap filter, compute frequency
% components in all time windows, integrate over two specific bands and
% train a classifier on those

clear
close all
% clc

% Load file
load('C:\Data\2017_07_MI_errP\S03\20170719T135827.mat')

% Lap filters (assuming here data were acquired with the 20 channels setup)
lapData=MI_session.applyLapFilter(obj.rawData.Data);

% Windows parameters
winStep=.2; % Approximate overlap, in seconds
winLength=1; % Window length, in seconds
winStarts=round((0:winStep:obj.currTime-winLength)*obj.fs);
ARmodelOrder=6; % Not sure how this impacts results
bandLims=[8,12,18,25]; % Limits of band of interest - i.e. 8-to-12 and 18-to-25
% relevantElectrodes=[1,7,11]; % i.e. Fz, C3 and C4
relevantElectrodes=1:20; % All of them
BP=zeros(length(winStarts),length(relevantElectrodes)*2); % Initialize band power matrix. Time points x (nChannes x nBands)
for currCh=1:length(relevantElectrodes)
    % Split data in windows
    relData=zeros(length(winStarts),winLength*obj.fs);
    for currWin=1:length(winStarts)
        relData(currWin,:)=lapData(winStarts(currWin)+1:winStarts(currWin)+obj.fs*winLength,relevantElectrodes(currCh));
    end
    % Compute power in each window using Yule-Walker PSD
    pxx=pyulear(relData',ARmodelOrder);
    
    % Bands of interest are 8-12 Hz and 18-25 Hz (from here: "Comparative
    % analysis of spectral approaches to feature extraction for EEG-based
    % motor imagery classification")
    binCenters=linspace(1/obj.fs,obj.fs/2,size(pxx,1));
    [~,bandStart(1)]=min(abs(binCenters-bandLims(1)));
    [~,bandEnd(1)]=min(abs(binCenters-bandLims(2)));
    [~,bandStart(2)]=min(abs(binCenters-bandLims(3)));
    [~,bandEnd(2)]=min(abs(binCenters-bandLims(4)));
    for currBand=1:2
        BP(:,(currCh-1)*2+currBand)=log(sum(pxx(bandStart(currBand):bandEnd(currBand),:)))';
    end
    fprintf('%d/%d\n',currCh,size(obj.rawData.Data,2));
end

% Get proper labels
lbls=zeros(length(obj.rawData.Data),1);
lbls(obj.fs*obj.MItimeStamps)=obj.trialLbls;
% lbls(obj.fs*obj.errPtimeStamps)=obj.fbLbls==obj.trialLbls;
B=ones(obj.fs,1);
lbls=filter(B,1,lbls);
lbls=lbls(winStarts+1);

% Data up to about 15000 samples is affected by starting artifact
affectedLength=round(15000/(obj.fs*winStep));
lbls(1:affectedLength)=[];
BP(1:affectedLength,:)=[];

svm=cell(max(lbls),1);
classEst=svm;
scores=svm;
AUC=zeros(max(lbls),1);
for currClass=1:max(lbls)
    % Trim data and labels so as to leave only instances of one class and
    % surrounding pauses (-1.5s, +1s). Use 1.4 instead to match winStep
    % multiple
    classStart=find(diff(lbls==currClass)==1);
    classEnd=find(diff(lbls==currClass)==-1);
    classStart=classStart-1.4/winStep;
    classEnd=classEnd+1/winStep;
    relSamples=[];
    for currTrial=1:length(classStart)
        relSamples=cat(2,relSamples,classStart(currTrial):classEnd(currTrial));
    end
    lblsShort=lbls(relSamples);
    BPShort=BP(relSamples,:);
    
    % Train a SVM for each class (excluding 0)
    C.NumTestSets=10;
    C.groups=ceil(linspace(1/length(lblsShort),C.NumTestSets,length(lblsShort)));
    %             C.groups=ceil(rand(size(lbls))*C.NumTestSets);
    C.training=@(currGroup)C.groups~=currGroup;
    C.test=@(currGroup)C.groups==currGroup;
    classEst{currClass}=zeros(length(lblsShort),1);
    scores{currClass}=zeros(length(lblsShort),2);
    for currP=1:C.NumTestSets
        % Recover training and testing sets
        trainData=BPShort(C.training(currP),:);
        testData=BPShort(C.test(currP),:);
        trainLbls=double(lblsShort(C.training(currP))==currClass);
        testLbls=lblsShort(C.test(currP))==currClass;
                
        % Perform actual training
        svm{currClass}=fitcsvm(trainData,trainLbls,'Standardize',true,'KernelScale','auto','KernelFunction','polynomial','PolynomialOrder',2);
        svm{currClass}=fitPosterior(svm{currClass});
        [classEst{currClass}(C.test(currP)),scores{currClass}(C.test(currP),:)]=predict(svm{currClass},testData);
    end
    [~,~,~,AUC(currClass)]=perfcurve(lblsShort,scores{currClass}(:,2),currClass);
end

% % Train a LDA 
% AUC=zeros(max(lbls),1);
% C=cvpartition(length(lbls),'kfold',10);
% classEst=zeros(length(lbls),1);
% scores=zeros(length(lbls),length(unique(lbls)));
% for currP=1:C.NumTestSets
%     % Recover training and testing sets
%     trainData=log(BP(C.training(currP),:));
%     testData=log(BP(C.test(currP),:));
%     trainLbls=double(lbls(C.training(currP)));
%     testLbls=lbls(C.test(currP));
%         
%     % Perform actual training
% %     lda=fitcdiscr(trainData,trainLbls);
% %     [classEst(C.test(currP)),scores(C.test(currP),:)]=predict(lda,testData);
% 
%     lda=fitcdiscr(trainData,trainLbls);
%     [classEst(C.test(currP)),scores(C.test(currP),:)]=predict(lda,testData);
% end

% % Train a SVM for each class (excluding 0)
% svm=cell(max(lbls),1);
% classEst=svm;
% scores=svm;
% AUC=zeros(max(lbls),1);
% C=cvpartition(length(lbls),'kfold',10);
% for currClass=1:max(lbls)
%     classEst{currClass}=zeros(length(lbls),1);
%     scores{currClass}=zeros(length(lbls),2);
%     for currP=1:C.NumTestSets
%         % Recover training and testing sets
%         trainData=log(BP(C.training(currP),:));
%         testData=log(BP(C.test(currP),:));
%         trainLbls=double(lbls(C.training(currP))==currClass);
%         testLbls=lbls(C.test(currP))==currClass;
%         
%         % Balance dataset
%         mostClassIdx=find(trainLbls==mode(trainLbls));
%         toBeDiscarded=mostClassIdx(randperm(length(mostClassIdx)-sum(trainLbls==setdiff(unique(trainLbls),mode(trainLbls)))));
%         trainData(toBeDiscarded,:)=[];
%         trainLbls(toBeDiscarded,:)=[];
%         
%         % Perform actual training
%         svm{currClass}=fitcsvm(trainData,trainLbls,'Standardize',true,'KernelScale','auto');
%         svm{currClass}=fitPosterior(svm{currClass});
%         [classEst{currClass}(C.test(currP)),scores{currClass}(C.test(currP),:)]=predict(svm{currClass},testData);
%     end
%     [~,~,~,AUC(currClass)]=perfcurve(lbls,scores{currClass}(:,2),1);
% end