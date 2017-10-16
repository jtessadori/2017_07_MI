clear
close all
clc

% Load results data
load('allSubjectsAUC.mat');

% 8 recordings on 6 subjects
subjID=ceil((1:length(AUC))/8);
subjIDcell=cell(length(AUC),1);
for currExp=1:length(AUC)
    subjIDcell{currExp}=sprintf('subj %d',subjID(currExp));
end

% Switch AUC in another format, too
AUC=cell2mat(AUC')';

% Groups
groups={'both hands','right hand','feet','left hand'};

% First boxplot
boxplot(AUC,groups);
xlabel('AUC');
title('AUC distributions (sorted by MI type)');
print('AUC_MItype','-dpng')

% Second boxplot
figure;
boxplot(AUC',subjIDcell);
xlabel('AUC');
title('AUC distributions (sorted by subject)');
print('AUC_subj','-dpng')

% % Prepare data to generate boxplot
% AUCvec=reshape(AUC,[],1);
% subjIDvec=[subjID,subjID+max(subjID),subjID+2*max(subjID),subjID+3*max(subjID)];
% 
% % Plot results
% boxplot(AUCvec,subjIDvec);