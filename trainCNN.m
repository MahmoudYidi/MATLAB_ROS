function trainCNN(varargin)
%% Dependencies
%  Local
addpath(genpath('./utils'));	% Utilities
addpath('./iterators');         % Training loops
addpath('./mdlgrads');          % Model gradient computers
addpath('./loss');              % Loss functions
addpath('./networks');          % Networks
addpath('./gfx');               % Graphics and plotting functions
addpath('./lrfinders');         % LR finder loops
addpath(genpath('./pose_fcns'));% Pose manipulation functions

%% Fix seed/other setup
rng('default');
rng(1);
clear settings;
delete(gcp('nocreate'));

%% Settings (user-modifiable)
%  Input
settings.imHeight = -1;                                   	% Resized input image height
settings.imWidth = -1;                                   	% Resized input image width
settings.imFile = ...                                       % Train and validation data filename (cell of sequences)
    '~/projects/datasets/city/oibar/docking-2021-02-18-stage2.mat';
settings.intMatFile = ...                                   % Intrinsic camera matrix file
    '~/projects/datasets/city/oibar/k_mat_rectified.txt';    
settings.isSeq = true;                                      % The selected data represents a sequence [true|false]
settings.modelFile = ...                                    % Previously trained model filename
	  '';
    
%  Training
settings.nChannelsIn = 3;                   % CNN number of image input channels
settings.nChannelsOut = 1024;               % CNN number of output channels
settings.dropoutCNN = 0.5;                  % CNN dropout factor
settings.numEpochs = 50;                    % Number of epochs
settings.miniBatchSize = 1;              	% Batch size
settings.lr = 5e-5;                         % Learning rate
settings.decay = 'triangular2';             % Learning rate decay
                                            %   ['none'|'linear'|'exponential'|'triangular'|'triangular2']
settings.decayEpoch = 0;                    % Epoch at which to start LR decay
settings.decayRate = 0.95;                  % Basis for exponential decay
settings.decaySteps = 5;                	% Exponential decay exponent denominator/cylical multiplier
settings.decayLrBound = 1e-3;               % Cyclical learning rate bound
settings.decayStaircase = false;         	% Smooth or discrete stepping [0|1]
settings.beta1 = 0.9;                       % Adam gradient decay factor
settings.beta2 = 0.999;                     % Adam squared gradient decay factor

%  Validation
settings.miniBatchSizeValid = 32;           % Validation batch size (can be larger for speed)

%  L2 Regularisation
settings.l2Reg.cnn.do = false;
settings.l2Reg.fc1.do = false;
settings.l2Reg.fc2.do = false;
settings.l2Reg.optw.do = false;

settings.l2Reg.cnn.lambda = 1e-4;
settings.l2Reg.fc1.lambda = 0.01;
settings.l2Reg.fc2.lambda = 0.01;
settings.l2Reg.optw.lambda = 0.0;

%  Learning rate finder
settings.lr_finder.minLr = 1e-10;   % Minimum learning rate for sweep
settings.lr_finder.maxLr = 100;     % Maximum learning rate for sweep
settings.lr_finder.epochs = 4;      % Number of epochs for sweep

%  Modes
settings.executionEnvironment = 'gpu';      % Training environment ['auto' | 'gpu']
settings.useParallel = false;               % Train network in parallel [true | false]
settings.runMode = 'train';                 % Running mode
                                            %   'train':          train network with set parameters
                                            %   'lr_finder':      learning rate finder sweep
                                            %   'lr_visualiser':  plot learning rate over iterations
settings.loadMode = {'none',...             % Model load mode
                        };                  %	'none': initialise all subnetworks
                                            %   'cnn':  load CNN
                                            %   'bn':   bridge network (between CNN and FCs)
                                            %   'fc1':  position FC layer
                                            %   'fc2':  attitude FC layer
                                            %   'optw': load adaptive weights
                                            %   'cont': pick-up training where left off
settings.trainMode = {'cnn','fc1', ...      % Model train mode
                      'fc2', 'optw'};       %   ['cnn'&'bn'&'fc1'&'fc2'&'optw']

%  Display
settings.epochLineMult = 10;      % Multiplier to display epoch vertical line in plots
settings.saveFigMult = 25;        % Global iteration multiplier to save figure

%  Image augmentation
settings.doImageAugmentation = true;
settings = augmentationOptions(settings, ...
                               'ChannelShift',                      true, ...
                               'ChannelShiftParams',                {0.0441}, ...
                               'GaussianBlur',                      true, ...
                               'GaussianBlurParams',                {0.0441, [3 9]}, ...
                               'GaussianNoise',                     true, ...
                               'GaussianNoiseParams',               {0.0441, [3e-3 1e-2]}, ...
                               'JpegCompression',                   true, ...
                               'JpegCompressionParams',             {0.0441, [1.5 2]}, ...
                               'MedianBlur',                        true, ...
                               'MedianBlurParams',                  {0.0441, [3 9]}, ...
                               'PatchDropout',                      true, ...
                               'PatchDropoutparams',                {0.0441, 0.15, [1 3]}, ...
                               'RandomBrightnessContrast',          true, ...
                               'RandomBrightnessContrastParams',    {0.0441}, ...
                               'RandomClahe',                       true, ...
                               'RandomClaheParams',                 {0.0441}, ...
                               'RandomGamma',                       true, ...
                               'RandomGammaParams',                 {0.0441, [50, 150]}, ...
                               'CamRotate',                         true, ...
                               'CamRotateParams',                   {0.4226, 20}, ...
                               'ImageRotate',                       true, ...
                               'ImageRotateParams',                 {0.4226, 170});

%% Settings (NON user-modifiable)
%  Get output folder
if (nargin == 0)
    settings.outFolder = 'output';
else
    settings.outFolder = varargin{1};
    mkdir(settings.outFolder);
end

%  Image base size setup
settings.imSize = [settings.imHeight settings.imWidth];

%  Processing mode setup
settings = modeSetup(settings);

%% Load training data
%  Images
load(settings.imFile,'imdsTrain','imdsValid');
if settings.isSeq
  imdsTrain = vertcat(imdsTrain{:});
end

% Intrinsic matrix
settings.kMat = readmatrix(settings.intMatFile);

%% Train DNN
switch (settings.runMode)
    case 'train'
        iterateCNN(settings, imdsTrain, imdsValid);
        
    case 'lr_finder'
        findLRCNN(settings, imdsTrain);
    
    case 'lr_visualiser'
        visualiseLR(settings, settings.miniBatchSize, imdsTrain, 'random');
        
    otherwise
        error('Invalid mode selected');
end

end
