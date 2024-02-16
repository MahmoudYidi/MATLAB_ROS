function [dlnetCNN,dlnetBN,dlnetFC1,dlnetFC2,dlparams,avgG,avgGS,globalIter,iniEpoch] = loadParams(settings)

% Subnetworks:
%   dlnetCNN	---     CNN backbone
%   dlnetBN     ---     Potential bridge network
%   dlnetFC     ---     Linear output
%   dlparams    ---     Position/attitude learnable weights
%
% Loading modes:
%   'none'      ---     Initialise all subnetworks
%   'cnn'       ---     Load dlnetCNN
%   'bn'        ---     Load dlnetBN
%   'fc'        ---     Load dlnetFC
%   'optw'      ---     Load dlparams
%   'cont'      ---     Continue training

% Load based on mode.
%   First, intialise all subnetworks and parameters with default values.
%   Subnetworks:
dlnetCNN    = squeezenetParams(settings);
dlnetBN     = bnParams();
dlnetFC1   	= fcParams(settings, settings.nChannelsOut, 3, false);    	% Position
dlnetFC2  	= fcParams(settings, settings.nChannelsOut, 6, false);      % Attitude
dlparams    = lossParams(settings);

%   Average gradients:
avgG.CNN    = [];	avgGS.CNN   = [];
avgG.BN     = [];	avgGS.BN    = [];
avgG.FC1    = [];	avgGS.FC1   = [];
avgG.FC2    = [];	avgGS.FC2   = [];
avgG.OptW   = [];	avgGS.OptW  = [];
%   Iterative indices:
globalIter  = 0;
iniEpoch    = 1;

%   Second, iterate through queried modes to see what needs to be loaded from file.
mode = settings.loadMode;
nMode = length(mode);
%   Check first if no subnets should be loaded.
if ~isempty(find(strcmp(mode, 'none')))
    return;
end
%   Otherwise, load requested subnets.
modelStruct = load(settings.modelFile);

for i = 1:nMode
    mode_i = mode{i};
    switch(mode_i)
        case 'cnn'
            dlnetCNN = modelStruct.dlnetCNN;
            avgG.CNN = modelStruct.avgG.CNN;
            avgGS.CNN = modelStruct.avgGS.CNN;
        case 'bn'
            dlnetBN = modelStruct.dlnetBN;
            avgG.BN = modelStruct.avgG.BN;
            avgGS.BN = modelStruct.avgGS.BN;
        case 'fc1'
            dlnetFC1 = modelStruct.dlnetFC1;
            avgG.FC1 = modelStruct.avgG.FC1;
            avgGS.FC1 = modelStruct.avgGS.FC1;
        case 'fc2'
            dlnetFC2 = modelStruct.dlnetFC2;
            avgG.FC2 = modelStruct.avgG.FC2;
            avgGS.FC2 = modelStruct.avgGS.FC2;
        case 'optw'
            dlparams = modelStruct.dlparams;
            avgG.OptW = modelStruct.avgG.OptW;
            avgGS.OptW = modelStruct.avgGS.OptW;
        case 'cont'
            globalIter = modelStruct.globalIter;
            iniEpoch = modelStruct.i + 1;
        otherwise
            error('Unknown mode');
    end
end

end
