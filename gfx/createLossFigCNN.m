function [handle,plotLines] = createLossFigCNN(iniEpoch,numIter)
% createLossFig
%
% Creates a figure to display the training and validation loss.
 
% set(0, 'CurrentFigure', plotHandle);
% clf reset;

% Setup
handle = figure;
plotLines = cell(6,1);                  % considering 6 different plots
colourTrain = [0, 0.4470, 0.7410];      % train loss line colour
colourValid = [0.8500, 0.3250, 0.0980]; % validation loss line colour
xLowerBound  = iniEpoch*numIter;

% Subplot 1 - general loss
subHandle = struct;

subHandle.ax = subplot(2,3,1);

subHandle.loss.train = animatedline('Color',colourTrain,'linewidth',1.2);
subHandle.loss.valid = animatedline('Color',colourValid,'linewidth',1.2,'marker','*','markersize',5);

xlim([xLowerBound inf]);
ylim([0 inf]);
xlabel("Iteration");
ylabel("Total Loss");
grid off;
grid minor;
box on;
legend([subHandle.loss.train subHandle.loss.valid],{'Training','Validation'},'Orientation','horizontal',...
    'Location','southoutside');

plotLines{1} = subHandle;

% Subplot 2 - zoom on general loss
subHandle = struct;

subHandle.ax = subplot(2,3,2);

subHandle.loss.train = animatedline('Color',colourTrain,'linewidth',1.2);
subHandle.loss.valid = animatedline('Color',colourValid,'linewidth',1.2,'marker','*','markersize',5);

xlim([xLowerBound inf]);
ylim([0 10]);
xlabel("Iteration");
ylabel("Loss (zoom)");
grid off;
grid minor;
box on;
legend([subHandle.loss.train subHandle.loss.valid],{'Training','Validation'},'Orientation','horizontal',...
    'Location','southoutside');

plotLines{2} = subHandle;

% Subplot 3 - translation error
subHandle = struct;

subHandle.ax = subplot(2,3,4);

subHandle.loss.train = animatedline('Color',colourTrain,'linewidth',1.2);
subHandle.loss.valid = animatedline('Color',colourValid,'linewidth',1.2,'marker','*','markersize',5);

xlim([xLowerBound inf]);
ylim([0 10]);
xlabel("Iteration");
ylabel("Position error (m)");
grid off;
grid minor;
box on;
legend([subHandle.loss.train subHandle.loss.valid],{'Training','Validation'},'Orientation','horizontal',...
    'Location','southoutside');

plotLines{3} = subHandle;

% Subplot 4 - rotation error
subHandle = struct;

subHandle.ax = subplot(2,3,5);

subHandle.loss.train = animatedline('Color',colourTrain,'linewidth',1.2);
subHandle.loss.valid = animatedline('Color',colourValid,'linewidth',1.2,'marker','*','markersize',5);

xlim([xLowerBound inf]);
ylim([0 25]);
xlabel("Iteration");
ylabel("Attitude error (deg)");
grid off;
grid minor;
box on;
legend([subHandle.loss.train subHandle.loss.valid],{'Training','Validation'},'Orientation','horizontal',...
    'Location','southoutside');

plotLines{4} = subHandle;

% Subplot 5 - learning rate
subHandle = struct;

subHandle.ax = subplot(2,3,3);

subHandle.lr = animatedline('Color','r','linewidth',1.2);

xlim([xLowerBound inf]);
ylim([0 inf]);
xlabel("Iteration");
ylabel("Learning rate");
grid off;
grid minor;
box on;

plotLines{5} = subHandle;

% Subplot 6 - adaptive weights
subHandle = struct;

subHandle.ax = subplot(2,3,6);

subHandle.sp = animatedline('Color','m','linewidth',1.2);
subHandle.sq = animatedline('Color','g','linewidth',1.2);

xlim([xLowerBound inf]);
ylim([-4 4]);
xlabel("Iteration");
ylabel("Weights");
grid off;
grid minor;
box on;
legend([subHandle.sp subHandle.sq],{'Position','Attitude'},'Orientation','horizontal',...
    'Location','southoutside');

plotLines{6} = subHandle;

%
set(handle, 'Units', 'centimeters', 'OuterPosition', [0, 0, 40, 25]);

end
