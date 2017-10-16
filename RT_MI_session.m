classdef RT_MI_session
    properties
        fs=512; % WARNING: DO NOT change this. Amplifier acquisition rate can only be changed from its panel in Simulink
        nClasses=2;
        timingTable;
        rawData;
        condition;
        trialParams;
        timingParams;
        figureParams;
        colorScheme;
        eventLog;
        modelName;
        timeTriggeredEvents;
        targetPos;
        trainedClassifier;
        outputLog;
    end
    properties (Dependent)
        recLength;
        currTime;
        currState;
    end
    properties (Hidden)
        possibleConditions={'Training','Testing'};
        isExpClosed=0;
    end
    methods
        %% Constructor
        function obj=RT_MI_session()
            % Define number of trials and position of pauses
            obj.trialParams.trialsPerSession=90;
            obj.trialParams.pausesPerSession=2;
            obj.trialParams.pausingTrials=linspace(1,obj.trialParams.trialsPerSession,obj.trialParams.pausesPerSession+2);
            obj.trialParams.pausingTrials=round(obj.trialParams.pausingTrials(2:end-1));
            
            % Set period length for different experiment parts
            obj.timingParams.cueMinLength=5;
            obj.timingParams.cueVarLength=2;
            obj.timingParams.waitMinLength=8;
            obj.timingParams.waitVarLength=2;
            
            % Set colors for different objects
            obj.colorScheme.bg=[.05,.05,.05];
            obj.colorScheme.targetColor=[0,.4,0];
            obj.colorScheme.cursorColor=[.4,0,.1];
            
            % Set possible positions for target
            obj.targetPos(1).X=[-.05,.05,.05,-.05];
            obj.targetPos(1).Y=[-.9,-.9,-.8,-.8];
            obj.targetPos(2).X=obj.targetPos(1).X;
            obj.targetPos(2).Y=[.8,.8,.9,.9];
            
            % Initialize a few things
            obj.eventLog.Times=[];
            obj.eventLog.Event=cell(0);
            obj.outputLog.actualState=[];
            obj.outputLog.estState=[];
            obj.outputLog.time=[];
            
            % Ask user whether to start experiment right away
            clc;
            if ~strcmpi(input('Start experiment now? [Y/n]\n','s'),'n')
                obj=runExperiment(obj);
            end
        end
        % Other methods
        function obj=runExperiment(obj)
            % Variables on base workspace will be used to trigger closing
            % of experiment
            assignin('base','isExpClosing',0);
            
            % Sets name of Simulink model to be used for acquisition
            obj.modelName='SimpleAcquisition_16ch_2014a_RT';
            
            % Prompts user to select a condition
            obj=selectCondition(obj);
            if obj.condition.conditionID==2
                obj.trainedClassifier=RT_MI_session.trainClassifier;
            end
            
            % Randomize trials errors
            obj=randomizeTrials(obj);
            
            % Prepares Simulink model (i.e. starts recording, basically)
            obj.prepareSimulinkModel;
            
            % Opens black figure as background
            obj=createExpFigure(obj);
            
            % Generates array of time triggered events
            obj.timeTriggeredEvents{1}=timeTriggeredEvent('updateScenario',0);
            obj.timeTriggeredEvents{2}=timeTriggeredEvent('updateCursor',0);
            
            % Shows a countdown
            obj.startCountdown(3);
            
            % Perform bulk of experiment
            obj=manageExperiment(obj);
            
            % Closes exp window and saves data
            obj.closeExp;
        end
        function obj=manageExperiment(obj)
            % Generate file name used to save experiment data
            fileName=datestr(now,30);
            while ~evalin('base','isExpClosing')&&obj.currTime<=obj.recLength
                pause(0.001);
                for currTTevent=1:length(obj.timeTriggeredEvents);
                    obj=checkAndExecute(obj.timeTriggeredEvents{currTTevent},obj.currTime,obj);
                    pause(0.001);
                end             
%                 if ismember(obj.currTrial,obj.pausingTrials)||(evalin('base','exist(''pauseNextTrial'',''var'')')&&evalin('base','pauseNextTrial'))
%                     assignin('base','pauseNextTrial',0);
%                     msgH=msgbox('This is a pause. Start again when ready.','Pause','modal');
%                     uiwait(msgH);
%                 end
            end
            obj.isExpClosed=1;
            delete(gcf);
            set_param(obj.modelName,'SimulationCommand','Stop');
            set_param(obj.modelName,'StartFcn','')
            obj.rawData=evalin('base','rawData');
            save(fileName,'obj');
            
            % Clear variables from base workspace
            evalin('base','clear listener*');
            evalin('base','clear errP*');
            evalin('base','clear MIdata*');
        end
        function obj=createExpFigure(obj)
            % Set figure properties
            obj.figureParams.handle=gcf;
            set(obj.figureParams.handle,'Tag',mfilename,...
                'Toolbar','none',...
                'MenuBar','none',...
                'Units','normalized',...
                'Resize','off',...
                'NumberTitle','off',...
                'Name','',...
                'Color',obj.colorScheme.bg,...
                'RendererMode','Manual',...
                'Renderer','OpenGL',...
                'WindowKeyPressFcn',@KeyPressed,...
                'CloseRequestFcn',@OnClosing,...
                'WindowButtonMotionFcn',@onMouseMove);
            
            % Plot target position
            obj.figureParams.target=patch(obj.targetPos(1).X,obj.targetPos(1).Y,obj.colorScheme.targetColor);
            obj.figureParams.cursor=patch(obj.targetPos(1).X,obj.targetPos(1).Y,obj.colorScheme.cursorColor);
            
            % Set and remove figure axis
            ylim([-1,1]);
            xlim([-1,1]);
            set(gcf,'units','normalized','position',[0,0,1,1]);
            axis square
            axis('off')
            
            % Remove box around figure
            undecorateFig;
        end
        function obj=updateScenario(obj)
            set(obj.figureParams.target,'XData',obj.targetPos(obj.currState).X,'YData',obj.targetPos(obj.currState).Y);
            obj.timeTriggeredEvents{1}.nextTrigger=obj.currTime+.01;
        end
        function obj=updateCursor(obj)
            if obj.condition.conditionID==2%&&evalin('base','exist(''currData'',''var'')')
                % Evaluate state using last available data
                dataWindow=evalin('base','currData');
                [~,currScores]=RT_MI_session.classifyWindow(dataWindow,obj.trainedClassifier);
                % Update cursor pos on screen
                movingRangeX=obj.targetPos(2).X-obj.targetPos(1).X;
                movingRangeY=obj.targetPos(2).Y-obj.targetPos(1).Y;
                set(obj.figureParams.cursor,'XData',obj.targetPos(1).X+movingRangeX*currScores(2),'YData',obj.targetPos(1).Y+movingRangeY*currScores(2));
                obj.outputLog.actualState=cat(1,obj.outputLog.actualState,obj.currState);
                obj.outputLog.estState=cat(1,obj.outputLog.estState,currScores(2));
                obj.outputLog.time=cat(1,obj.outputLog.time,obj.currTime);
            end
            % Fix next evaluation time for this function
            obj.timeTriggeredEvents{1}.nextTrigger=obj.currTime+.05;
        end
        function obj=selectCondition(obj)
            clc;
            for currPossibleCond=1:length(obj.possibleConditions)
                fprintf('[%d] - %s;\n',currPossibleCond,obj.possibleConditions{currPossibleCond});
            end
            currCond=input('\nPlease select desired condition: ');
            obj.condition.conditionID=currCond;
        end
        function obj=randomizeTrials(obj)
            % Produce list of movements
            cueLengths=rand(obj.trialParams.trialsPerSession)*obj.timingParams.cueVarLength+obj.timingParams.cueMinLength;
            waitLengths=rand(obj.trialParams.trialsPerSession+1)*obj.timingParams.waitVarLength+obj.timingParams.waitMinLength;
            obj.timingTable=[waitLengths(1),0];
            for currChange=1:obj.trialParams.trialsPerSession
                obj.timingTable=cat(1,obj.timingTable,[obj.timingTable(end,1)+cueLengths(currChange),1]);
                obj.timingTable=cat(1,obj.timingTable,[obj.timingTable(end,1)+waitLengths(currChange),0]);
            end
        end
        function prepareSimulinkModel(obj)
            % Check whether simulink model file can be found
            if ~exist(obj.modelName,'file')
                warning('Cannot find model %s.\nPress Enter to continue.\n',obj.modelName);
                input('');
                [fileName,pathName]=uigetfile('*.slx','Select Simulink model to load:');
                obj.modelName=sprintf('%s\\%s',pathName,fileName);
            end
            % Load model
            load_system(obj.modelName);
            
            % Check whether simulation was already running, and, in case,
            % stop it
            if bdIsLoaded(obj.modelName)&&strcmp(get_param(obj.modelName,'SimulationStatus'),'running')
                set_param(obj.modelName,'SimulationCommand','Stop');
            end
            
            % Add event listener to triggered buffer event.
            set_param(obj.modelName,'StartFcn',sprintf('simulinkModelStartFcn(''%s'')',obj.modelName))
            set_param(obj.modelName,'StopTime','inf');
            set_param(obj.modelName,'FixedStep',['1/',num2str(obj.fs)]);
            set_param(obj.modelName,'SimulationCommand','Start');
        end
        function wait(obj,pauseLength)
            startTime=get_param(obj.modelName,'SimulationTime');
            while strcmp(get_param(obj.modelName,'SimulationStatus'),'running')&&get_param(obj.modelName,'SimulationTime')<=startTime+pauseLength
                pause(1/(2*obj.fs));
            end
        end
        function startCountdown(obj,nSecs)
            % countdown to experiment start
            figure(obj.figureParams.handle)
            for cntDown=nSecs:-1:1
                if ~exist('textHandle','var')
                    textHandle=text(-.05,.5,num2str(cntDown));
                else
                    set(textHandle,'String',num2str(cntDown));
                end
                set(textHandle,'Color','white','FontSize',64);
                pause(1);
            end
            delete(textHandle);
        end
        %% Dependent properties
        function rl=get.recLength(obj)
            rl=sum(obj.timingTable(:,1));
        end
        function cTime=get.currTime(obj)
            if obj.isExpClosed
                cTime=obj.rawData.Time(end);
            else
                cTime=get_param(obj.modelName,'SimulationTime');
            end
        end
        function cs=get.currState(obj)
            cs=obj.timingTable(sum(obj.currTime>obj.timingTable(:,1))+1,2)+1;
        end
    end
    methods (Static)
        function closeExp
            % Signals experiment to close
            assignin('base','isExpClosing',1);
        end
        function [currClass,currScores]=classifyWindow(dataWindow,trainedClassifier)
            BP=zeros(1,trainedClassifier.nChannels*2); % Initialize band power matrix.
            lapData=dataWindow*trainedClassifier.fltrWeights;
            for currCh=1:trainedClassifier.nChannels
                % Compute power in input window using Yule-Walker PSD
                pxx=pyulear(lapData(:,currCh),trainedClassifier.ARmodelOrder);
                
                % Compute power in bands of interest
                binCenters=linspace(1/trainedClassifier.fs,trainedClassifier.fs/2,size(pxx,1));
                [~,bandStart(1)]=min(abs(binCenters-trainedClassifier.bandLims(1)));
                [~,bandEnd(1)]=min(abs(binCenters-trainedClassifier.bandLims(2)));
                [~,bandStart(2)]=min(abs(binCenters-trainedClassifier.bandLims(3)));
                [~,bandEnd(2)]=min(abs(binCenters-trainedClassifier.bandLims(4)));
                for currBand=1:2
                    BP((currCh-1)*2+currBand)=log(sum(pxx(bandStart(currBand):bandEnd(currBand),:)))';
                end
            end
            [currClass,currScores]=predict(trainedClassifier.classifier,BP);
        end
        function [trainedClassifier,AUC]=trainClassifier(varargin)
            % If no argument is passed, user is prompted to select a
            % recording file. Default parameters are use to perform
            % analysis on selected file. If a path to a file is provided as
            % an argument, analysis is performed on indicated file. If a
            % structure is provided as argument, it is assumed to contain
            % classifier parameters that override default.
            persistent pathName
            
            % Check for presence of file
            if nargin>0
                for currArg=1:nargin
                    if isa(varargin{currArg},'char')&&exist(varargin{currArg},'file')
                        load(varargin{currArg});
                        noFileFound=0;
                        break;
                    end
                    noFileFound=1;
                end
            end
            if nargin==0||noFileFound
                if isempty(pathName)||(isa(pathName,'double')&&pathName==0)
                    pathName=pwd;
                end
                [fileName,pathName]=uigetfile(pathName,'Select file to load:');
                if fileName==0
                    warning('No file selected, closing');
                    trainedClassifier=[];
                    return;
                end
                load(sprintf('%s%s',pathName,fileName));
            end
            
            % Default parameters
            trainedClassifierDefault.fs=obj.fs;
            trainedClassifierDefault.ARmodelOrder=6; % Not sure how this impacts results
            trainedClassifierDefault.bandLims=[8,12,18,25]; % Limits of band of interest - i.e. 8-to-12 and 18-to-25
%             trainedClassifierDefault.bandLims=ceil((1:.5:24.5)); % Limits of band of interest - single Hertz bands from 1 to 25
            trainedClassifierDefault.winStep=.5; % Step, in seconds
            trainedClassifierDefault.winLength=1; % Window length, in seconds
            trainedClassifierDefault.nChannels=size(obj.rawData.Data,2);
            trainedClassifierFields={'fs','ARmodelOrder','bandLims','winStep','winLength','nChannels'};
            
            % Check for presence of struct overriding default parameters
            if nargin>0
                for currArg=1:nargin
                    if isa(varargin{currArg},'struct')
                        trainedClassifier=varargin{1};
                        trainedClassifier.isDefault=0;
                        break;
                    end
                trainedClassifier.isDefault=1;
                end
            else
                trainedClassifier.isDefault=1;
            end
            for currField=1:length(trainedClassifierFields)
                if ~isfield(trainedClassifier,trainedClassifierFields{currField})
                    trainedClassifier.(trainedClassifierFields{currField})=trainedClassifierDefault.(trainedClassifierFields{currField});
                end
            end            
            
            % Lap filters (assuming here data were acquired with the 20 channels setup)
            [lapData,trainedClassifier.fltrWeights]=RT_MI_session.applyLapFilter(obj.rawData.Data);
            
% %             Time filter
%             [B,A]=butter(4,2/(obj.fs/2));
%             lapData=filter(B,A,lapData);
    
            winStarts=round((0:trainedClassifier.winStep:obj.currTime-trainedClassifier.winLength)*trainedClassifier.fs);
            BP=zeros(length(winStarts),size(obj.rawData.Data,2)*length(trainedClassifier.bandLims)/2); % Initialize band power matrix. Time points x (nChannes x nBands)
            for currCh=1:size(obj.rawData.Data,2)
                % Split data in windows
                relData=zeros(length(winStarts),trainedClassifier.winLength*obj.fs);
                for currWin=1:length(winStarts)
                    relData(currWin,:)=lapData(winStarts(currWin)+1:winStarts(currWin)+obj.fs*trainedClassifier.winLength,currCh);
                end
                
                % Compute power in each window using Yule-Walker PSD
                matLabVersion=version;
                if matLabVersion(1)=='9' % Previous versions of matlab did not support the use of pyulear on matrices
                    pxx=pyulear(relData',trainedClassifier.ARmodelOrder);
                else
                    pxx=zeros(obj.fs/4+1,length(winStarts));
                    for currWin=1:length(winStarts)
                        pxx(:,currWin)=pyulear(relData(currWin,:),trainedClassifier.ARmodelOrder);
                    end
                end
                
                % Bands of interest are 8-12 Hz and 18-25 Hz (from here: "Comparative
                % analysis of spectral approaches to feature extraction for EEG-based
                % motor imagery classification")
                binCenters=linspace(1/obj.fs,obj.fs/2,size(pxx,1));
                for currBand=1:length(trainedClassifier.bandLims)/2
                    [~,bandStart]=min(abs(binCenters-trainedClassifier.bandLims(currBand*2-1)));
                    [~,bandEnd]=min(abs(binCenters-trainedClassifier.bandLims(currBand*2)));
                    BP(:,(currCh-1)*length(trainedClassifier.bandLims)/2+currBand)=log(sum(pxx(bandStart:bandEnd,:)))';
                end
                fprintf('%d/%d\n',currCh,size(obj.rawData.Data,2));
            end
            feats=BP;
            
            % Get proper labels
            lbls=zeros(length(winStarts),1);
            for currWin=1:length(lbls)
                lbls(currWin)=obj.timingTable(sum((winStarts(currWin)/obj.fs)>obj.timingTable(:,1))+1,2);
            end
            
            % Skip first samples to limit amount of initial artifact 
            lbls(1:round(60/trainedClassifier.winStep))=[];
            feats(1:round(60/trainedClassifier.winStep),:)=[];
            
            % Train a SVM for each class (excluding 0)
%             C=cvpartition(length(lbls),'kfold',10);
            C.NumTestSets=4;
            C.groups=ceil(linspace(1/length(lbls),C.NumTestSets,length(lbls)));
%             C.groups=ceil(rand(size(lbls))*C.NumTestSets);
            C.training=@(currGroup)C.groups~=currGroup;
            C.test=@(currGroup)C.groups==currGroup;
            classEst=zeros(length(lbls),1);
            scores=zeros(length(lbls),2);
            for currP=1:C.NumTestSets
                % Recover training and testing sets
                trainData=feats(C.training(currP),:);
                testData=feats(C.test(currP),:);
                trainLbls=double(lbls(C.training(currP)));
                
                % Perform actual training
                svm=fitcsvm(trainData,trainLbls,'Standardize',true,'KernelScale','auto','KernelFunction','polynomial','PolynomialOrder',2);
                svm=fitPosterior(svm);
                [classEst(C.test(currP)),scores(C.test(currP),:)]=predict(svm,testData);
            end
            [~,~,~,AUC]=perfcurve(lbls,scores(:,2),1);
            disp(AUC);
            
            % Perform actual training
            subspaceDimension=max(1,min(60,size(feats,2)-1));
            trainedClassifier.classifier=fitensemble(feats,lbls,'Subspace',30,'KNN','Type','Classification','NPredToSample',subspaceDimension);
%             trainedClassifier.classifier=fitcsvm(feats,lbls,'Standardize',true,'KernelScale','auto','KernelFunction','polynomial','PolynomialOrder',2);
%             trainedClassifier.classifier=fitPosterior(trainedClassifier.classifier);
        end
        function testClassifier
            persistent pathName
            if isempty(pathName)||(isa(pathName,'double')&&pathName==0)
                pathName=pwd;
            end
            [fileName,pathName]=uigetfile(pathName,'Select data file to analyze:');
            if fileName==0
                warning('No file selected, closing');
                return;
            end
            load(sprintf('%s%s',pathName,fileName));
                        
            if isempty(pathName)||(isa(pathName,'double')&&pathName==0)
                pathName=pwd;
            end
            [fileName,pathName]=uigetfile(pathName,'Select file containing classifier to test (cancel to use classifier in loaded data):');
            if fileName~=0
                load(sprintf('%s%s',pathName,fileName));
                obj.trainedClassifier=trainedClassifier;
            end
            
            scores=zeros(length(obj.outputLog.time),2);
            for currWin=1:length(scores)            
                winStart=(obj.outputLog.time(currWin)-1)*obj.fs+1;
                winEnd=obj.outputLog.time(currWin)*obj.fs;
                [~,scores(currWin,:)]=RT_MI_session.classifyWindow(obj.rawData.Data(winStart:winEnd,:),obj.trainedClassifier);
            end
            [~,~,~,AUC]=perfcurve(obj.outputLog.actualState,scores(:,2),2);
            disp(AUC);
        end
        function [outData,fltrWeights]=applyLapFilter(inData)
            try
                load('elMap16.mat')
            catch ME %#ok<NASGU>
                warning('''elMap.mat'' not found. Electrode map required for laplacian filters.');
                outData=[];
                return;
            end
            fltrWeights=zeros(size(inData,2));
            for currEl=1:size(inData,2)
                neighborsMap=zeros(size(elMap16.elMat));
                neighborsMap(elMap16.elMat==currEl)=1;
                neighborsMap=imdilate(neighborsMap,strel('diamond',1));
                neighborsMap(elMap16.elMat==currEl)=0;
                validNeighbors=logical(neighborsMap.*elMap16.elMat);
                fltrWeights(currEl,elMap16.elMat(validNeighbors))=-1/sum(sum(validNeighbors));
                fltrWeights(currEl,currEl)=1;
            end
            outData=inData*fltrWeights;
        end
    end
end

function simulinkModelStartFcn(modelName) %#ok<DEFNU>
% Start function for Simulink model.
blockName=sprintf('%s/triggeredBuffer/Buffer',modelName);
assignin('base','listener',add_exec_event_listener(blockName,'PostOutputs',@acquireBufferedData));
end

function acquireBufferedData(block,~)
assignin('base','currData',block.OutputPort(1).Data);
end

function onMouseMove(~,~)
% Makes mouse pointer invisible
if ~strcmp(get(gcf,'Pointer'),'custom')
    set(gcf,'PointerShapeCData',NaN(16));
    set(gcf,'Pointer','custom');
end
end

function KeyPressed(~,eventdata,~)
% This is called each time a keyboard key is pressed while the mouse cursor
% is within the window figure area
if strcmp(eventdata.Key,'escape')
    RT_MI_session.closeExp;
end
if strcmp(eventdata.Key,'p')
    keyboard;
    %     assignin('base','pauseNextTrial',1)
end
end

function OnClosing(~,~)
% Overrides normal closing procedure so that regardless of how figure is
% closed logged data is not lost
RT_MI_session.closeExp;
end