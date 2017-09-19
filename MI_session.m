classdef MI_session
    properties
        fs=512;
        trialsPerSession
        nClasses=4; % Number of possible different actions to distinguish (rest state is not included)
        trialLbls; % Vector with labels for each trial, as shown during cue
        rawData;
        eventLog;
        timingParams; % Time length of different trial sections, in seconds
        condition;
        fakeVisualFeedbackCorrectChance=.6; % Probability of visual feedback being of correct class
        fbLbls; % Vectors with labels for each trial, as shown during feedback
        elMap;
    end
    properties (Dependent)
        modelName; % Char string with name of Simulink model to launch on experiment startup
        currTime;
        MItimeStamps;
        errPtimeStamps;
    end
    properties (Hidden)
        possibleConditions={'Visual feedback','Visuo-tactile feedback'};
        mName=[];
        figureParams; % Struct with information on figure parameters
        currTrial=1; % Next movement number
        colorScheme;
        pausesPerSession;
        pausingTrials;
        motorSerialPort;
        isExpClosed=0;
        elMapHidden;
    end
    methods
        %% Constructor
        function obj=MI_session()
            % Define number of trials and position of pauses
            obj.trialsPerSession=90;
            obj.pausesPerSession=2;
            obj.pausingTrials=linspace(1,obj.trialsPerSession,obj.pausesPerSession+2);
            obj.pausingTrials=round(obj.pausingTrials(2:end-1));
            
            % Set period length for different experiment parts
            obj.timingParams.fixCross=1.5;
            obj.timingParams.cue=4;
            obj.timingParams.FB=2;
            obj.timingParams.minWait=2;
            obj.timingParams.maxRandWait=.5;
            
            % Set colors for different objects
            obj.colorScheme.bg=[.05,.05,.05];
            obj.colorScheme.edgeColor=[.4,.4,.4];
            obj.colorScheme.cueFill=[0,.3,0];
            obj.colorScheme.cursorFill=[.4,0,.1];
            
            % Initialize a few things
            obj.eventLog.Times=[];
            obj.eventLog.Event=cell(0);
            
            % Ask user whether to start experiment right away
            clc;
            if ~strcmpi(input('Start experiment now? [Y/n]\n','s'),'n')
                obj=runExperiment(obj);
            end
        end
        % Dependent properties
        function obj=set.modelName(obj,newName)
            obj.mName=newName;
            % Put name of simulink model in variable 'modelName' in base
            % workspace
            assignin('base','modelName',newName);
        end
        function newName=get.modelName(obj)
            newName=obj.mName;
        end
        function cTime=get.currTime(obj)
            if obj.isExpClosed
                cTime=obj.rawData.Time(end);
            else
                if bdIsLoaded(obj.modelName)&&strcmp(get_param(obj.modelName,'SimulationStatus'),'running')
                    cTime=get_param(obj.modelName,'SimulationTime');
                else
                    cTime=toc;
                end
            end
        end
        function timeStamps=get.errPtimeStamps(obj)
            timeStamps=obj.eventLog.Times(cellfun(@(x)numel(strfind(x,'drawFB'))>0,obj.eventLog.Event));
        end
        function timeStamps=get.MItimeStamps(obj)
            timeStamps=obj.eventLog.Times(cellfun(@(x)numel(strfind(x,'fillCue'))>0,obj.eventLog.Event));
        end
        function elMap=get.elMap(obj) %#ok<MANU>
            if exist('elMap20.mat','file')
                load elMap20.mat
                elMap=elMap20;
            else
                elMap=[];
            end
        end
        %% Other functions
        function obj=runExperiment(obj)
            % Variables on base workspace will be used to trigger closing
            % of experiment
            assignin('base','isExpClosing',0);
            
            % Sets name of Simulink model to be used for acquisition
            obj.modelName='SimpleAcquisition_20ch_2014a';
            
            % Prompts user to select a condition
            obj=selectCondition(obj);
            
            % Start motors
            obj=initializeMotors(obj);
            
            % Randomize trials errors
            obj=randomizeTrials(obj);
            
            % Prepares Simulink model (i.e. starts recording, basically)
            obj.prepareSimulinkModel;
            
            % Opens black figure as background
            obj=createExpFigure(obj);
            
            % Shows a countdown
            obj.startCountdown(3);
            tic; % Start a timer, in case simulink model is not running
            
            % Perform bulk of experiment
            obj=manageExperiment(obj);
            
            % Closes exp window and saves data
            obj.closeExp;
        end
        function obj=manageExperiment(obj)
            % Generate file name used to save experiment data
            fileName=datestr(now,30);
            while ~evalin('base','isExpClosing')&&obj.currTrial<=obj.trialsPerSession
                obj=drawCross(obj);
                wait(obj,obj.timingParams.fixCross);
                obj=fillCue(obj);
                wait(obj,obj.timingParams.cue);
                obj=clearCue(obj);
                wait(obj,1);
                obj=drawFB(obj);
                wait(obj,obj.timingParams.FB);
                obj=hidePatches(obj);
                wait(obj,obj.timingParams.minWait+rand*obj.timingParams.maxRandWait);
                
                % Save data up to now, in case something goes wrong.
                % WARNING: cannot save EEG data from here, need to stop
                % model first
%                 evalin('base','save temp.mat');
                                
                % Increase counter
                obj.currTrial=obj.currTrial+1;
                
                if ismember(obj.currTrial,obj.pausingTrials)||(evalin('base','exist(''pauseNextTrial'',''var'')')&&evalin('base','pauseNextTrial'))
                    assignin('base','pauseNextTrial',0);
                    msgH=msgbox('This is a pause. Start again when ready.','Pause','modal');
                    uiwait(msgH);
                end
            end
            obj.isExpClosed=1;
            delete(gcf);
            if obj.condition.conditionID==2
                fclose(obj.motorSerialPort);
            end
            if bdIsLoaded(obj.modelName)&&strcmp(get_param(obj.modelName,'SimulationStatus'),'running')
                set_param(obj.modelName,'SimulationCommand','Stop');
                obj.rawData=evalin('base','rawData');
            end
            save(fileName,'obj');
            
            % Clear variables from base workspace
            evalin('base','clear listener*');
            evalin('base','clear errP*');
            evalin('base','clear MIdata*');
        end
        function obj=drawCross(obj)
            set(obj.figureParams.cueHandle,{'EdgeColor'},{obj.colorScheme.edgeColor;obj.colorScheme.edgeColor;obj.colorScheme.edgeColor;obj.colorScheme.edgeColor});
            obj.figureParams.crossHandle(1)=line([-.1,.1],[0,0],'Color',obj.colorScheme.edgeColor,'LineWidth',2);
            obj.figureParams.crossHandle(2)=line([0,0],[-.1,.1],'Color',obj.colorScheme.edgeColor,'LineWidth',2);
            obj.figureParams.cursorHandle=rectangle('Position',[-.06,-.06,.12,.12],'Curvature',[1,1],'EdgeColor',obj.colorScheme.edgeColor,'FaceColor',obj.colorScheme.cursorFill);
            obj=obj.updateLog;
        end
        function obj=fillCue(obj)
            if obj.condition.conditionID==2
                obj.sendMotorCommand(obj.trialLbls(obj.currTrial));
                % Prepare correct vibration length for next step
                fprintf(obj.motorSerialPort,'T10');
            end
            set(obj.figureParams.cueHandle(obj.trialLbls(obj.currTrial)),'FaceColor',obj.colorScheme.cueFill);
            obj=obj.updateLog;
        end
        function obj=drawFB(obj)
            switch obj.fbLbls(obj.currTrial)
                case 1
                    xPos=0;
                    yPos=.3;
                case 2
                    xPos=.3;
                    yPos=0;
                case 3
                    xPos=0;
                    yPos=-.3;
                case 4
                    xPos=-.3;
                    yPos=0;
            end
            set(obj.figureParams.cursorHandle,'Position',[xPos-.06,yPos-.06,.12,.12]);
            if obj.condition.conditionID==2
                obj.sendMotorCommand(obj.fbLbls(obj.currTrial));
                % Prepare correct vibration length for next step
                fprintf(obj.motorSerialPort,['T',num2str(round(obj.timingParams.cue)*10)]);
            end
            obj=obj.updateLog;
        end
        function obj=clearCue(obj)
            set(obj.figureParams.cueHandle,{'FaceColor'},{obj.colorScheme.bg;obj.colorScheme.bg;obj.colorScheme.bg;obj.colorScheme.bg});
            obj=obj.updateLog;
        end
        function obj=hidePatches(obj)
            set(obj.figureParams.cueHandle,{'EdgeColor'},{obj.colorScheme.bg;obj.colorScheme.bg;obj.colorScheme.bg;obj.colorScheme.bg});
            delete(obj.figureParams.crossHandle(1));
            delete(obj.figureParams.crossHandle(2));
            delete(obj.figureParams.cursorHandle);
            obj=obj.updateLog;
        end
        function obj=updateLog(obj)
            drawnow;
            st=dbstack;
            obj.eventLog.Times=cat(1,obj.eventLog.Times,obj.currTime);
            obj.eventLog.Event=cat(1,obj.eventLog.Event,st(2).name);
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
            obj.trialLbls=linspace(1,obj.nClasses-.0001+1,obj.trialsPerSession);
            obj.trialLbls=floor(obj.trialLbls);
            obj.trialLbls=obj.trialLbls(randperm(obj.trialsPerSession));
            obj.fbLbls=randi(obj.nClasses,size(obj.trialLbls,1),size(obj.trialLbls,2));
            correctChance=rand(1,obj.trialsPerSession)<(obj.fakeVisualFeedbackCorrectChance*obj.nClasses-1)/(obj.nClasses-1); % Trust me on this one... or check it, but it's correct
            obj.fbLbls(correctChance)=obj.trialLbls(correctChance);
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
            
            % Load Simulink model, sets its start function and launch it
            set_param(obj.modelName,'StopTime','inf');
            set_param(obj.modelName,'FixedStep',['1/',num2str(obj.fs)]);
            set_param(obj.modelName,'SimulationCommand','Start');
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
            
            % Plot different objects and keep them hidden by painting them
            % as background
            obj.figureParams.cueHandle(1)=patch([-.05,-.05,-.1,0,.1,.05,.05],[.6,.65,.65,.75,.65,.65,.6],obj.colorScheme.bg,'EdgeColor',obj.colorScheme.bg);
            obj.figureParams.cueHandle(2)=patch([.6,.65,.65,.75,.65,.65,.6],[-.05,-.05,-.1,0,.1,.05,.05],obj.colorScheme.bg,'EdgeColor',obj.colorScheme.bg);
            obj.figureParams.cueHandle(3)=patch([-.05,-.05,-.1,0,.1,.05,.05],[-.6,-.65,-.65,-.75,-.65,-.65,-.6],obj.colorScheme.bg,'EdgeColor',obj.colorScheme.bg);
            obj.figureParams.cueHandle(4)=patch([-.6,-.65,-.65,-.75,-.65,-.65,-.6],[-.05,-.05,-.1,0,.1,.05,.05],obj.colorScheme.bg,'EdgeColor',obj.colorScheme.bg);
            
            % Set and remove figure axis
            set(gcf,'units','normalized','position',[0,0,1,1]);
            axis square
            axis('off')
            
            % Remove box around figure
            undecorateFig;
        end
        function wait(obj,pauseLength)
            if bdIsLoaded(obj.modelName)&&strcmp(get_param(obj.modelName,'SimulationStatus'),'running')
                startTime=get_param(obj.modelName,'SimulationTime');
                while strcmp(get_param(obj.modelName,'SimulationStatus'),'running')&&get_param(obj.modelName,'SimulationTime')<=startTime+pauseLength
                    pause(1/(2*obj.fs));
                end
            else
                pause(pauseLength);
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
        function obj=initializeMotors(obj)
            if obj.condition.conditionID==2
                % Open serial port to control vibration motors
                obj.motorSerialPort=serial('COM5','BaudRate',250000);
                fopen(obj.motorSerialPort);
                
                % Pause
                pause(1)
                
                % Need to send any command before it can work
                fprintf(obj.motorSerialPort,'\n');
                
                % Set default vibration length to long vibration
                fprintf(obj.motorSerialPort,['T',num2str(round(obj.timingParams.cue-2)*10)]);
            end
        end
        function obj=sendMotorCommand(obj,currCommand)
            % Send movement  a vibration on motors.
            switch currCommand
                case 1
                    motorCommand='W';
                case 2
                    motorCommand='2';
                case 3
                    motorCommand='A';
                case 4
                    motorCommand='1';
            end
            fprintf(obj.motorSerialPort,motorCommand);
        end
        %% Results analysis and plotting
        function plotErrPs(obj)
            freqData=freqFilter(obj,obj.rawData.data);
            CARdata=MI_session.CARfilter(freqData);
            timeWins=MI_session.splitData(obj.errPtimeStamps*obj.fs,CARdata,obj.fs);
            errs=1-(obj.trialLbls==obj.fbLbls);
            for currCh=1:size(freqData,2)
                subplot(4,5,currCh);
                plot(linspace(0,1,obj.fs),squeeze(median(timeWins(errs==0,:,currCh))));
                hold on
                plot(linspace(0,1,obj.fs),squeeze(median(timeWins(errs==1,:,currCh))),'r');
                plot(linspace(0,1,obj.fs),squeeze(median(timeWins(errs==1,:,currCh)))-squeeze(median(timeWins(errs==0,:,currCh))),'g');
                if ~isempty(obj.elMap)
                    xlabel(obj.elMap.elName{currCh});
                end
                axis([0,1,-10,10]);
            end
        end
        function trndClassifier=trainErrP(obj)
%             trimmedData=trimData(obj.rawData.data);
            freqData=MI_session.freqFilter(obj.rawData.data,obj.fs);
            [CARdata,coeffs]=MI_session.CARfilter(freqData);
            shortData=MI_session.downsampleData(CARdata,obj.fs);
            timeWins=MI_session.splitData(round(obj.errPtimeStamps*64),shortData,64);
            timeWinsLong=MI_session.splitData(round(obj.errPtimeStamps*obj.fs),CARdata,obj.fs);
            fftData=MI_session.recoverFrequencyData(timeWinsLong);
            errs=1-(obj.trialLbls==obj.fbLbls);
            feats=cat(2,fftData,reshape(timeWins,length(errs),[]));
%             feats=reshape(timeWins,length(errs),[]);
            
            r=.4;
            cv=cvpartition(length(errs),'leaveout');
            classEst=zeros(size(errs));
            scores=zeros(length(errs),2);
            for currP=1:cv.NumTestSets
                % Recover training and testing sets and labels
                trainData=feats(cv.training(currP),:);
                testData=feats(cv.test(currP),:);
                trainLbls=errs(cv.training(currP));
                
                % Perform classification
                classifier=fitcsvm(trainData,trainLbls,'KernelScale','auto','Standardize',true,'Cost',[0,r;1-r,0]);
                [classEst(cv.test(currP)),scores(cv.test(currP),:)]=predict(classifier,testData);
            end
            [~,~,~,AUC]=perfcurve(errs,scores(:,1),0);
            accs=[sum((classEst==0).*(errs==0))/sum(errs==0),sum((classEst==1).*(errs==1))/sum(errs==1)];
            
            fprintf('Mean acc: %0.2f %0.2f\nAUC: %0.2f\ncurrent r: %0.2f\n',accs(1),accs(2),AUC,r)
            
            % Train final classifier
            trndClassifier.classifier=fitcsvm(feats,errs,'KernelScale','auto','Standardize',true,'Cost',[0,r;1-r,0]);
            trndClassifier.coeffs=coeffs;
        end
        function plotMIs(obj)
%             lapData=applyLapFilter(obj.rawData.data);
%             timeFreq=bankFilter(obj,lapData);
%             keyboard;
            freqData=freqFilter(obj,obj.rawData.data);
            lapData=applyLapFilter(freqData);
            timeWinsLong=MI_session.splitData(round(obj.MItimeStamps*obj.fs),lapData,obj.fs*4);
            for currCh=1:size(freqData,2)
                subplot(4,5,currCh);
                plot(linspace(0,4,obj.fs*4),squeeze(median(timeWinsLong(obj.trialLbls==1,:,currCh))));
                hold on
                plot(linspace(0,4,obj.fs*4),squeeze(median(timeWinsLong(obj.trialLbls==2,:,currCh))),'r');
                plot(linspace(0,4,obj.fs*4),squeeze(median(timeWinsLong(obj.trialLbls==3,:,currCh))),'g');
                plot(linspace(0,4,obj.fs*4),squeeze(median(timeWinsLong(obj.trialLbls==4,:,currCh))),'k');
            end
        end
        function trndClassifier=trainMI(obj)
            freqData=freqFilter(obj,obj.rawData.data);
            lapData=applyLapFilter(freqData);
            shortData=downsampleData(lapData,obj.fs);
            timeWins=MI_session.splitData(round(obj.MItimeStamps*64),shortData,64*4);
            timeWinsLong=MI_session.splitData(round(obj.MItimeStamps*obj.fs),lapData,obj.fs*4);
            fftData=recoverFrequencyData(timeWinsLong);
            lbls=obj.trialLbls;
            feats=cat(2,fftData,reshape(timeWins,length(lbls),[]));
            
%                         classifier=fitcdiscr(feats,lbls);
%             coeffsSum=zeros(size(classifier.Coeffs(1,2).Linear));
%             for c1=1:4
%                 for c2=1:4
%                     if c1~=c2
%                         coeffsSum=coeffsSum+abs(classifier.Coeffs(c1,c2).Linear);
%                     end
%                 end
%             end
%             [~,coeffsOrdr]=sort(coeffsSum,'descend');
%             relFeats=coeffsOrdr(1:20);
            
            cv=cvpartition(length(lbls),'leaveout');
            classEst=zeros(size(lbls));
            scores=zeros(length(lbls),length(unique(lbls)));
            for currP=1:cv.NumTestSets
                % Recover training and testing sets and labels
                trainData=feats(cv.training(currP),:);
                testData=feats(cv.test(currP),:);
                trainLbls=lbls(cv.training(currP));
                
                % Perform classification
                classifier=fitcdiscr(trainData,trainLbls);
                coeffsSum=zeros(size(classifier.Coeffs(1,2).Linear));
                for c1=1:4
                    for c2=1:4
                        if c1~=c2
                            coeffsSum=coeffsSum+abs(classifier.Coeffs(c1,c2).Linear);
                        end
                    end
                end
                [~,coeffsOrdr]=sort(coeffsSum,'descend');
                relFeats=coeffsOrdr(1:40);
                lblsMat=zeros(length(trainLbls),length(unique(lbls)));
                for currClass=1:4
                    lblsMat(trainLbls==currClass,currClass)=1;
                end
                [A,~,~,U]=canoncorr(trainData(:,relFeats),lblsMat);
                classifier=fitcdiscr(U,trainLbls);
                U=(testData(:,relFeats)-mean(testData(:,relFeats)))*A;
                [classEst(cv.test(currP)),scores(cv.test(currP),:)]=predict(classifier,U);
            end
%                 
%                 
%                 classifier=fitcsvm(trainData,trainLbls,'KernelScale','auto','Standardize',true);
%                 [classEst(cv.test(currP)),scores(cv.test(currP),:)]=predict(classifier,testData);
%             end
%             [~,~,~,AUC]=perfcurve(lbls,scores(:,1),0);
            AUC=0;
            for currClass=1:length(unique(lbls))
                accs(currClass)=sum((classEst==currClass).*(lbls==currClass))/sum(lbls==currClass); %#ok<AGROW>
            end
            
            fprintf('Mean acc: %0.2f %0.2f\nAUC: %0.2f\n\n',accs(1),accs(2),AUC)
            
            trndClassifier.classifier=classifier;
%             % Train final classifier
%             trndClassifier.classifier=fitcsvm(feats,lbls,'KernelScale','auto','Standardize',true,'Cost',[0,r;1-r,0]);
%             trndClassifier.coeffs=coeffs;
        end
        function outData=bankFilter(obj,inData)
            [B,A]=butter(4,[1,32]/(obj.fs/2));
            freqData=zeros(size(inData));
            for currCh=1:size(freqData,2)
                freqData(:,currCh)=filtfilt(B,A,inData(:,currCh));
            end
            timeFreq=zeros(size(freqData,1),size(freqData,2),29);
            for currFilt=2:30
                [B,A]=butter(4,[currFilt-1,currFilt]/(obj.fs/2));
                for currCh=1:size(freqData,2)
                    timeFreq(:,currCh,currFilt-1)=filtfilt(B,A,freqData(:,currCh));
                end
            end
            timeFreq=timeFreq.^2;
            outData=zeros(size(timeFreq));
            for currFilt=1:29
                if currFilt==29
                    keyboard;
                end
                B=blackman(ceil(256/currFilt));
                for currCh=1:size(freqData,2)
                    outData(:,currCh,currFilt)=filtfilt(B,1,timeFreq(:,currCh,currFilt));
                end
            end
        end
    end
    methods (Static)
        function closeExp
            % Signals experiment to close
            assignin('base','isExpClosing',1);
        end
        %% Preprocessing
        function freqData=freqFilter(inData,fs,varargin)
            % If two arguments are passed, band-pass between 1 and 10 Hz.
            % If a third argument (vector of two real numbers) is passed,
            % use those as limits of band-pass filter
            if nargin>2
                freqLims=varargin{1};
            else
                freqLims=[1,10];
            end            
            [B,A]=butter(4,freqLims/(fs/2));
            freqData=zeros(size(inData));
            for currCh=1:size(freqData,2)
                freqData(:,currCh)=filter(B,A,inData(:,currCh));
            end
        end
        function [CARdata,coeff]=CARfilter(inData)
            CARdata=zeros(size(inData));
            coeff=zeros(1,size(inData,2));
            for currCh=1:size(inData,2)
                otherChsMedian=median(inData(:,[1:currCh-1,currCh+1:end]),2);
                coeff(currCh)=pinv(otherChsMedian)*inData(:,currCh);
                CARdata(:,currCh)=inData(:,currCh)-otherChsMedian*coeff(currCh);
            end
        end
        function timeWins=splitData(timeStamps,inData,winLength)
            timeWins=zeros(length(timeStamps),winLength,size(inData,2));
            for currTimeStamp=1:length(timeStamps)
                timeWins(currTimeStamp,:,:)=inData(timeStamps(currTimeStamp)+1:timeStamps(currTimeStamp)+winLength,:);
            end
        end
        function shortData=downsampleData(inData,fs)
            shortData=resample(inData,64,fs);
        end
        function fftData=recoverFrequencyData(inData)
            fftData=fft(inData.*repmat(blackman(size(inData,2))',size(inData,1),1,size(inData,3)),[],2);
            fftData=fftData(:,2:20,:);
            fftData=reshape([real(fftData),imag(fftData)],size(fftData,1),[]);
        end
        function outData=applyLapFilter(inData)
            try
                load('elMap20.mat')
            catch ME %#ok<NASGU>
                warning('''elMap20.mat'' not found. Electrode map required for laplacian filters.');
                outData=[];
                return;
            end
            fltrWeights=zeros(size(inData,2));
            for currEl=1:size(inData,2)
                neighborsMap=zeros(size(elMap20.elMat));
                neighborsMap(elMap20.elMat==currEl)=1;
                neighborsMap=imdilate(neighborsMap,strel('diamond',1));
                neighborsMap(elMap20.elMat==currEl)=0;
                validNeighbors=logical(neighborsMap.*elMap20.elMat);
                fltrWeights(currEl,elMap20.elMat(validNeighbors))=-1/sum(sum(validNeighbors));
                fltrWeights(currEl,currEl)=1;
            end
            outData=inData*fltrWeights;
        end
    end
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
    MI_session.closeExp;
end
if strcmp(eventdata.Key,'p')
    keyboard;
%     assignin('base','pauseNextTrial',1)
end
end

function OnClosing(~,~)
% Overrides normal closing procedure so that regardless of how figure is
% closed logged data is not lost
MI_session.closeExp;
end
