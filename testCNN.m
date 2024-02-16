function testCNN(X)
%% Dependencies
%  Local
addpath(genpath('./utils'));	% Utilities
addpath('./networks');          % Networks
addpath(genpath('./pose_fcns'));% Pose manipulation functions

%% Fix seed/other setup
clear settings;

%% Settings (user-modifiable)
%  Input
settings.imFile = ...                                       % Data filename
    './sampledata';
settings.modelFile = ...                                    % Previously trained model filename
	  './results/old/model45.mat';
  
%  Runtime
settings.executionEnvironment = 'gpu';      % Execution environment ['auto' | 'gpu']
settings.useParallel = false;               % Train network in parallel [true | false]

%% Settings (NON user-modifiable)
%  Processing mode setup
settings = modeSetup(settings);

%% Load data
%  Images
%imdsTest = imageDatastore(settings.imFile);

%  Network weights

persistent modelStruct;
  if isempty(modelStruct)
     modelStruct = load(settings.modelFile);
  end

dlnetCNN = modelStruct.dlnetCNN;

%clear modelStruct

%% Testing environment
%nImgs = length(imdsTest.Files); 

%  Main loop
    X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
    
    % Convert data to dlarray specify the dimension
    % labels 'SSC' (spatial, spatial, channel).
    dlX = dlProcess(X,'SSC',settings);
    
    % Run network.
    dlY = squeezenetModel(dlX, dlnetCNN, 'test');

    % Process results.
    dlY_t = dlY(1:3);
    dlY_sixd = dlY(4:9);

    t_hat = dlUnprocess(dlY_t,settings);
    sixd_hat = dlUnprocess(dlY_sixd,settings);

    r_hat = sixdimToRotmat(sixd_hat);
    q_hat = so3_to_su2(r_hat);

    % Display result.
    % [t1 t2 t3 q1 q2 q3 q4]
    fprintf("%0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f\n", ...
        t_hat(1),t_hat(2),t_hat(3),...
        q_hat(1),q_hat(2),q_hat(3),q_hat(4));


end
