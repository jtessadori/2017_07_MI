classdef MI_speed_control
    properties
        fs=512; % WARNING: DO NOT change this. Amplifier acquisition rate can only be changed from its panel in Simulink
        nTargets;
        rawData;
        cursorPos;
        targetPos;
        condition;
        figureParams;
        colorScheme;
        modelName;
        timeTriggeredEvents;
        linearMap;
        outputLog;
        recLength
        trainingParams;
        bufferData;
    end
    properties (Dependent)
        currTime;
        currStage;
        currTrial;
    end
    properties (Hidden)
        CDlength;
        possibleConditions={'1D training, horz','1D testing, horz'};
        isExpClosed=0;
        isTraining=0;
        forceTrials=0;
        isCursorVisible;
        actualTarget;
        lastStage=0;        
    end
    methods
        %% Constructor
        function obj=MI_speed_control(varargin)
            % If an argument is passed, it must be a structure matching
            % linearMap template (i.e. basically, an output of another
            % session). Some parameters (e.g. sampling frequency of
            % amplifier and buffer window length) cannot be changed
            % directly from here, so please do make sure that they're
            % matching with the current settings of relevant Simulink
            % model.
            
            % Set length of initial countdown, in seconds (first ~2 minutes
            % of recording are affected by hw filter oscillations)
            obj.CDlength=15;
            
            % Set desired length of recording. Will be overwritten during
            % training (after last trial, recording closes)
            obj.recLength=295;
            
            % Set colors for different objects
            obj.colorScheme.bg=[.05,.05,.05];
            obj.colorScheme.targetColor=[0,.4,0];
            obj.colorScheme.cursorColor=[.4,0,.1];
            obj.colorScheme.fixColor=[.4,.4,.4];
            
            % Set shape and pos for cursor
            obj.figureParams.cursorShape.X=[-.05,.05,.05,-.05];
            obj.figureParams.cursorShape.Y=[-.05,-.05,.05,.05];
            obj.cursorPos=[0,0];
            
            % Set shape for fixation cross
            obj.figureParams.fixCross(1).X=[-.1,.1];
            obj.figureParams.fixCross(1).Y=[0,0];
            obj.figureParams.fixCross(2).X=[0,0];
            obj.figureParams.fixCross(2).Y=[-.1,.1];
            
            % Set possible target centers and shape
            % Two targets, horz
            obj.targetPos{1}=[-.9,0];
            obj.targetPos{2}=[.9,0];
% %             N-targets
%             targetAngles=linspace(0,2*pi,11);
%             for currTarget=1:length(targetAngles)-1
%                 obj.targetPos{currTarget}(1)=.9*cos(targetAngles(currTarget));
%                 obj.targetPos{currTarget}(2)=.9*sin(targetAngles(currTarget));
%             end
            obj.figureParams.targetRadius=.1;
            obj.figureParams.targetShape.X=cos(linspace(0,2*pi,40))*obj.figureParams.targetRadius;
            obj.figureParams.targetShape.Y=sin(linspace(0,2*pi,40))*obj.figureParams.targetRadius;
            obj.nTargets=length(obj.targetPos);
            
            % Training parameters, if used
            obj.trainingParams.fixCross=1.5;
            obj.trainingParams.cue=4;
            obj.trainingParams.minITI=2;
            obj.trainingParams.maxITI=3;
            obj.trainingParams.nTrials=25;
            
            % Default parameters
            if nargin==0
                linearMapDefault.fs=obj.fs;
                linearMapDefault.ARmodelOrder=14; % From: "Noninvasive Electroencephalogram Based Control of a Robotic Arm for Reach and Grasp Tasks"
                %             linearMapDefault.bandLims=[8,12,18,25]; % Limits of band of interest - i.e. 8-to-12 and 18-to-25
                %             linearMapDefault.bandLims=ceil((1:.5:24.5)); % Limits of band of interest - single Hertz bands from 1 to 25
                linearMapDefault.bandLims=[10,14]; % Cited work
                % WARNING: the following line DOESN'T DO ANYTHING; it is just a
                % reference. Buffer length has to be changed in the Simulink
                % model (specifically, in the triggeredBuffer mask
                % initialization)
                linearMapDefault.winLength=.4; % Window length, in seconds. Also, cited work
                linearMapDefault.relChannels=1:16; % Use all channels
%                 linearMapDefault.relChannels=[7,11]; % C3 and C4 in 16 el setup
                linearMapDefault.nChannels=length(linearMapDefault.relChannels);
                obj.linearMap=linearMapDefault;
            else
                obj.linearMap=varargin{1};
            end
            
            % Initialize a few things
            obj.outputLog.time=[];
            obj.outputLog.cursorSpeed=[];
            obj.outputLog.cursorPos=[];
            obj.outputLog.actualTarget=[];
            obj.outputLog.feats=[];
            obj.outputLog.isTraining=[];
            obj.outputLog.stage=[];
            obj.linearMap.feats=cell(obj.nTargets,1);
            for currTrgt=1:obj.nTargets
%                 obj.linearMap.feats{currTrgt}=zeros(1,obj.linearMap.nChannels*length(obj.linearMap.bandLims)/2);
                obj.linearMap.classMedian{currTrgt}=zeros(1,obj.linearMap.nChannels*length(obj.linearMap.bandLims)/2);
                obj.linearMap.classVar{currTrgt}=zeros(1,obj.linearMap.nChannels*length(obj.linearMap.bandLims)/2);
            end
            obj.outputLog.targetsReached.time=[];
            obj.outputLog.targetsReached.correctTarget=[];
            obj.outputLog.targetsReached.targetID=[];
            
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
            obj=setConditionSpecificParams(obj);

            % Randomly select a target
            if obj.currTrial==1
                obj.actualTarget=obj.trainingParams.targetSequence(obj.currTrial);
            else
                obj.actualTarget=ceil(rand*obj.nTargets);
            end
            
            % Prepares Simulink model (i.e. starts recording, basically)
            obj.prepareSimulinkModel;
            
            % Opens black figure as background
            obj=createExpFigure(obj);
            
            % Generates array of time triggered events
            obj.timeTriggeredEvents{1}=timeTriggeredEvent('updateScenario',0);
            obj.timeTriggeredEvents{2}=timeTriggeredEvent('updateCursorEvent',0);
            obj.timeTriggeredEvents{3}=timeTriggeredEvent('toggleTraining',0);
            
            % Shows a countdown
            obj.startCountdown(obj.CDlength);
            
            % Perform bulk of experiment
            obj=manageExperiment(obj);
            
            % Closes exp window and saves data
            obj.closeExp;
        end
        
        function obj=manageExperiment(obj)
            % Generate file name used to save experiment data
            fileName=datestr(now,30);
            while ~evalin('base','isExpClosing')&&obj.currTime<=(obj.recLength+obj.CDlength)
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
            evalin('base','clear toggleTraining');
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
            if ~obj.forceTrials
                obj.figureParams.target=patch(obj.figureParams.targetShape.X+obj.targetPos{obj.actualTarget}(1),obj.figureParams.targetShape.Y+obj.targetPos{obj.actualTarget}(2),obj.colorScheme.targetColor);
            end
            if obj.isCursorVisible
                obj.figureParams.cursor=patch(obj.figureParams.cursorShape.X+obj.cursorPos(1),obj.figureParams.cursorShape.Y+obj.cursorPos(2),obj.colorScheme.cursorColor);
            end
            
            % Set and remove figure axis
            ylim([-1,1]);
            xlim([-1,1]);
            set(gcf,'units','normalized','position',[0,0,1,1]);
            axis square
            axis('off')
            
            % Remove box around figure
%             undecorateFig;
        end
        
        function obj=updateScenario(obj)
            if obj.forceTrials
                % Check whether scenario needs updating
                if obj.currStage>obj.lastStage
                    obj.lastStage=obj.currStage;
                    switch obj.trainingParams.timeTable(obj.currStage,2)
                        case 0
                            if isfield(obj.figureParams,'crossHandle')&&ishandle(obj.figureParams.crossHandle(1))
                                delete(obj.figureParams.crossHandle(1));
                                delete(obj.figureParams.crossHandle(2));
                            end
                            if isfield(obj.figureParams,'target')&&ishandle(obj.figureParams.target)
                                delete(obj.figureParams.target);
                            end
                        case 1
                            obj.figureParams.crossHandle(1)=line(obj.figureParams.fixCross(1).X,obj.figureParams.fixCross(1).Y,'Color',obj.colorScheme.fixColor,'LineWidth',2);
                            obj.figureParams.crossHandle(2)=line(obj.figureParams.fixCross(2).X,obj.figureParams.fixCross(2).Y,'Color',obj.colorScheme.fixColor,'LineWidth',2);
                        case 2
                            obj.actualTarget=obj.trainingParams.targetSequence(obj.currTrial);
                            obj.figureParams.target=patch(obj.figureParams.targetShape.X+obj.targetPos{obj.actualTarget}(1),obj.figureParams.targetShape.Y+obj.targetPos{obj.actualTarget}(2),obj.colorScheme.targetColor);
                    end
                end
            else
                % If I am in free driving, only changing part of scenario
                % is position of target
                set(obj.figureParams.target,'XData',obj.figureParams.targetShape.X+obj.targetPos{obj.actualTarget}(1),'YData',obj.figureParams.targetShape.Y+obj.targetPos{obj.actualTarget}(2));
            end
            drawnow;
            
            % Set next evaluation time for this function
            obj.timeTriggeredEvents{1}.triggersLog=[obj.timeTriggeredEvents{1}.triggersLog,obj.currTime];
            obj.timeTriggeredEvents{1}.nextTrigger=obj.currTime+.01;
        end
        
        function obj=updateCursorEvent(obj)
            % Recover data buffer from base workspace (Simulink puts them
            % there)
            dataWindow=evalin('base','currData');
%             dataTimeStamp=obj.currTime;
            
%             % Update normalization buffer and normalize data
%             obj=updateBufferData(obj,dataWindow,dataTimeStamp);
%             fixDim=@(x)repmat(x,size(dataWindow,1),1);
%             if obj.bufferData.SD>0 % Skip first window to prevent Infs
%                 dataWindow=(dataWindow-fixDim(obj.bufferData.mean))./fixDim(obj.bufferData.SD);
%             end
            
            % If this is first iteration, compute laplacian filter weights
            if ~isfield(obj.linearMap,'lapFltrWeights')
                [~,obj.linearMap.lapFltrWeights]=MI_speed_control.applyLapFilter(dataWindow);
            end
            
            % Recover bandpower data from data buffer
            BP=MI_speed_control.preprocData(dataWindow,obj.linearMap);
            
            % Next line over-writes actual data for testing purposes.
            % Should be commented during actual use
%             BP=zeros(size(BP));
%             BP(1:2)=obj.targetPos{obj.actualTarget};
            
            % If training is ongoing, update linear map
            if obj.isTraining
                obj=obj.computeLinearMap(BP);
            end
            
            % Evaluate current cursor speed using last available data, if
            % at least one training step has been completed
            if isfield(obj.linearMap,'mat')
                cursorSpeed=MI_speed_control.computeSpeed(BP,obj.linearMap);
                
                % If cursor position is being displayed, update it
                if obj.isCursorVisible
%                     % Add small help during training
%                     if obj.isTraining
%                         cursorSpeed=cursorSpeed+0.01*sign(obj.targetPos{obj.actualTarget}-obj.cursorPos);
%                     end
                    % Update cursor position
                    obj=obj.updateCursorPos(cursorSpeed);
                end
            else
                cursorSpeed=[0,0];
            end
            drawnow;
            
            % Add relevant info to log
            obj.outputLog.cursorSpeed=cat(1,obj.outputLog.cursorSpeed,cursorSpeed);
            obj.outputLog.cursorPos=cat(1,obj.outputLog.cursorPos,obj.cursorPos);
            obj.outputLog.actualTarget=cat(1,obj.outputLog.actualTarget,obj.actualTarget);
            obj.outputLog.feats=cat(1,obj.outputLog.feats,BP);
            obj.outputLog.isTraining=cat(1,obj.outputLog.isTraining,obj.isTraining);
            obj.outputLog.time=cat(1,obj.outputLog.time,obj.currTime);
            if isfield(obj.trainingParams,'timeTable')
                obj.outputLog.stage=cat(1,obj.outputLog.stage,obj.trainingParams.timeTable(obj.currStage,2));
            end
            
            % Check if target is reached and change it, in case
            for currTarget=1:obj.nTargets
                if sqrt(sum((obj.cursorPos-obj.targetPos{currTarget}).^2))<obj.figureParams.targetRadius
                    obj.cursorPos=[0,0];
                    obj.outputLog.targetsReached.time=cat(1,obj.outputLog.targetsReached.time,obj.currTime);
                    obj.outputLog.targetsReached.correctTarget=cat(1,obj.outputLog.targetsReached.correctTarget,currTarget==obj.actualTarget);
                    obj.outputLog.targetsReached.targetID=cat(1,obj.outputLog.targetsReached.correctTarget,currTarget);
                    obj.actualTarget=ceil(rand*obj.nTargets);
                    break;
                end
            end
            
            % Set next evaluation time for this function
            obj.timeTriggeredEvents{2}.triggersLog=[obj.timeTriggeredEvents{2}.triggersLog,obj.currTime];
            obj.timeTriggeredEvents{2}.nextTrigger=obj.currTime+.05;
        end
        
        function obj=updateBufferData(obj,dataWindow,dataTimeStamp)
            % At the moment, I am only taking a single time point for each
            % dataWindow to estimate mean and sd in the last thirty
            % seconds. This is VERY approximate, better ideas for a
            % different solution are welcome
            if isfield(obj.bufferData,'data')
                obj.bufferData.data=cat(1,obj.bufferData.data,dataWindow(1,:));
                obj.bufferData.timeStamps=cat(1,obj.bufferData.timeStamps,dataTimeStamp);
            else
                obj.bufferData.data=dataWindow(1,:);
                obj.bufferData.timeStamps=dataTimeStamp;
            end
            toBeRemoved=obj.currTime-obj.bufferData.timeStamps>30;
            obj.bufferData.data(toBeRemoved,:)=[];
            obj.bufferData.timeStamps(toBeRemoved)=[];
            obj.bufferData.mean=mean(obj.bufferData.data,1);
            obj.bufferData.SD=std(obj.bufferData.data,[],1);
        end
        
        function obj=computeLinearMap(obj,BP)
            % Initialize all feats classes with the first window analyzed
            % (otherwise wouldn't be able to compute other stuff until a
            % sample per class was acquired)
            if ~isfield(obj.linearMap,'feats')
                for currTarget=1:obj.nTargets
                    obj.linearMap.feats{currTarget}=BP;
                end
            else
                obj.linearMap.feats{obj.actualTarget}=cat(1,obj.linearMap.feats{obj.actualTarget},BP);
            end
            
            % Compute median of current class
            obj.linearMap.classMedian{obj.actualTarget}=median(obj.linearMap.feats{obj.actualTarget},1);
            
            % Coefficients of linear map are computed so that ideally a
            % fixed number of iterations are required to reach each target
            % from start position
            obj.linearMap.mat=((cell2mat(obj.targetPos')'/50)*pinv(cat(2,cell2mat(obj.linearMap.classMedian'),ones(obj.nTargets,1))'))';
%             x=cat(2,cell2mat(obj.linearMap.classMedian'),ones(obj.nTargets,1));
%             y=cell2mat(obj.targetPos')/50;
%             for currDim=1:2
%                 [B,fitInfo]=lasso(x,y(:,currDim),'Alpha',.5);
%                 obj.linearMap.mat(:,currDim)=B(:,fitInfo.MSE==min(fitInfo.MSE));
%             end
            
            % Estimate variance from median to reduce impact of outliers
            obj.linearMap.classVar{obj.actualTarget}=(1/0.6745*median(abs(obj.linearMap.feats{obj.actualTarget}-repmat(obj.linearMap.classMedian{obj.actualTarget},size(obj.linearMap.feats{obj.actualTarget},1),1)))).^2;
        end
        
        function obj=toggleTraining(obj)
            if evalin('base','exist(''toggleTraining'',''var'')')&&evalin('base','toggleTraining')
                obj.isTraining=~obj.isTraining;
                assignin('base','toggleTraining',0);
                figure(obj.figureParams.handle)
                if obj.isTraining
                    textHandle=text(-.4,.5,'Training on');
                else
                    textHandle=text(-.4,.5,'Training off');
                end
                set(textHandle,'Color','white','FontSize',64);
            	wait(obj,.5);
                delete(textHandle);
            end
            
            % Set next evaluation time for this function
            obj.timeTriggeredEvents{3}.triggersLog=[obj.timeTriggeredEvents{3}.triggersLog,obj.currTime];
            obj.timeTriggeredEvents{3}.nextTrigger=obj.currTime+.5;
        end
        
        function obj=selectCondition(obj)
            clc;
            for currPossibleCond=1:length(obj.possibleConditions)
                fprintf('[%d] - %s;\n',currPossibleCond,obj.possibleConditions{currPossibleCond});
            end
            currCond=input('\nPlease select desired condition: ');
            obj.condition.conditionID=currCond;
        end
        
        function obj=setConditionSpecificParams(obj)
            % '1D training, horz','1D testing, horz'
            switch obj.condition.conditionID
                case 1
                    obj.isTraining=1;
                    obj.isCursorVisible=0;
                    obj.forceTrials=1;
                    randomPauses=rand(obj.trainingParams.nTrials,1)*(obj.trainingParams.maxITI-obj.trainingParams.minITI);
                    obj.trainingParams.targetSequence=ceil(rand(obj.trainingParams.nTrials,1)*obj.nTargets);
                    % 0, black screen, 1 fix cross on, 2 cue on
                    obj.trainingParams.timeTable=[];
                    for currTrial=1:obj.trainingParams.nTrials
                        obj.trainingParams.timeTable=cat(1,obj.trainingParams.timeTable,[randomPauses(currTrial)+obj.trainingParams.minITI,0]);
                        obj.trainingParams.timeTable=cat(1,obj.trainingParams.timeTable,[obj.trainingParams.fixCross,1]);
                        obj.trainingParams.timeTable=cat(1,obj.trainingParams.timeTable,[obj.trainingParams.cue,2]);
                    end
                    obj.trainingParams.timeTable=cat(1,obj.trainingParams.timeTable,[3,0]); % Add a 3 s pause at the end before closing recording
                    obj.trainingParams.timeTable(:,3)=cumsum(obj.trainingParams.timeTable(:,1));
                    obj.recLength=obj.trainingParams.timeTable(end,3);
                    obj.trainingParams.timeTable(:,3)=obj.trainingParams.timeTable(:,3)+obj.CDlength;
                case 2
                    obj.isTraining=0;
                    obj.isCursorVisible=1;
                    obj.forceTrials=0;
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
        
        function obj=updateCursorPos(obj,cursorSpeed)
            % Limit speed in borders to prevent cursors from getting out of
            % screen
            expectedTargetPos=obj.cursorPos+cursorSpeed;
            for currDim=1:length(cursorSpeed)
                overshoot=abs(expectedTargetPos(currDim))-0.9;
                if overshoot>0
                    newOvershoot=atan(overshoot*10*pi/2)/(10*pi/2);
                    obj.cursorPos(currDim)=(.9+newOvershoot)*sign(expectedTargetPos(currDim));
                else
                    obj.cursorPos(currDim)=expectedTargetPos(currDim);
                end
            end
            if obj.isCursorVisible
                set(obj.figureParams.cursor,'XData',obj.figureParams.cursorShape.X+obj.cursorPos(1),'YData',obj.figureParams.cursorShape.Y+obj.cursorPos(2));
            end
        end
        
        %% Dependent properties
        function cTime=get.currTime(obj)
            if obj.isExpClosed
                cTime=obj.rawData.Time(end);
            else
                cTime=get_param(obj.modelName,'SimulationTime');
            end
        end
        
        function cStage=get.currStage(obj)
            if isfield(obj.trainingParams,'timeTable')
                cStage=sum(obj.currTime>obj.trainingParams.timeTable(:,3))+1;
                cStage=min(cStage,size(obj.trainingParams.timeTable,1));
            else
                cStage=0;
            end
        end
        
        function cTrial=get.currTrial(obj)
            if isfield(obj.trainingParams,'timeTable')
                cTrial=ceil(obj.currStage/3);
            else
                cTrial=0;
            end
        end
        
    end
    methods (Static)
        function cursorSpeed=computeSpeed(BP,linearMap)
            %                 cursorSpeed=[mean((linearMap.Q-BP).*linearMap.featWeights),0];
%             cursorSpeed=[0,0];
%             for currTarget=1:length(linearMap.vec)
%                 if ~isempty(linearMap.vec{currTarget})
%                     cursorSpeed=cursorSpeed+BP*linearMap.vec{currTarget};
%                 end
%             end
            cursorSpeed=[BP,1]*linearMap.mat;
            if sum(~isfinite(cursorSpeed)) % Prevent weird speeds on startup
                cursorSpeed=[0,0];
            end
            % Cap speed around .05
            corrFactor=(pi/2)/.05;
            cursorSpeed=atan(cursorSpeed*corrFactor)/corrFactor;
        end
        
        function preprocessedData=preprocData(dataWindow,linearMap)
            preprocessedData=zeros(1,linearMap.nChannels*length(linearMap.bandLims)/2); % Initialize band power matrix.
            lapData=dataWindow*linearMap.lapFltrWeights;
            for currCh=1:linearMap.nChannels
                % Compute power in input window using Yule-Walker PSD
                pxx=pyulear(detrend(lapData(:,currCh)),linearMap.ARmodelOrder);
                
                % Compute power in bands of interest
                binCenters=linspace(1/linearMap.fs,linearMap.fs/2,size(pxx,1));
                for currBand=1:length(linearMap.bandLims)/2
                    [~,bandStart]=min(abs(binCenters-linearMap.bandLims(currBand*2-1)));
                    [~,bandEnd]=min(abs(binCenters-linearMap.bandLims(currBand*2)));
                    preprocessedData(:,(currCh-1)*length(linearMap.bandLims)/2+currBand)=sum(pxx(bandStart:bandEnd,:))';
                end
            end
            % The following line switches from absolute to relative power
            % levels ACROSS CHANNELS
            preprocessedData=preprocessedData/sum(preprocessedData,2);
            preprocessedData=log(preprocessedData);
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
        
        function closeExp
            % Signals experiment to close
            assignin('base','isExpClosing',1);
        end
        
%         function spectrogramData=computeSpectrogram(obj)
%             fltrdData=MI_speed_control.applyLapFilter(obj.rawData.Data);
%             winSampleLength=round(obj.linearMap.winLength*obj.fs);
%             winStarts=1:winSampleLength:length(fltrdData)-winSampleLength;
%             spectrogram=zeros(129,length(winStarts),size(obj.rawData.Data,2));
%             fprintf('This will take some time...\n');
%             for currWin=1:length(winStarts)
%                 for currCh=1:size(obj.rawData.Data,2)
%                     spectrogram(:,currWin,currCh)=pyulear(detrend(fltrdData(winStarts(currWin):winStarts(currWin)+winSampleLength,currCh)),obj.linearMap.ARmodelOrder);
%                 end
%                 fprintf('%d\\%d\n',currWin,length(winStarts));
%             end
%             spectrogramData.spctr=spectrogram;
%             spectrogramData.times=winStarts*obj.linearMap.winLength;
%         end
%         
%         function plotSpectrogram(spectrogramData)
%             for currCh=1:size(spectrogramData.spectrogram,2)
%                 figure;
%                 imagesc(log(squeeze(spectrogramData.spectrogram(:,:,1)))')
%             end
%         end
    end
end

function simulinkModelStartFcn(modelName) %#ok<DEFNU>
% Start function for Simulink model.
blockName=sprintf('%s/triggeredBuffer/Buffer',modelName);
assignin('base','listener',add_exec_event_listener(blockName,'PostOutputs',@acquireBufferedData));
end

function acquireBufferedData(block,~)
assignin('base','currData',block.OutputPort(1).Data);
assignin('base','currTime',block.SampleTime);
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
if strcmp(eventdata.Key,'t')
    assignin('base','toggleTraining',1);
end
end

function OnClosing(~,~)
% Overrides normal closing procedure so that regardless of how figure is
% closed logged data is not lost
RT_MI_session.closeExp;
end