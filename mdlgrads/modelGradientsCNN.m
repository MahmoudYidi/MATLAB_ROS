function [loss, errPos, errSo3, sp, sq, varargout] = ...
    modelGradientsCNN(dlX, dlYTrue, dlnetCNN, dlnetBN, dlnetFC1, dlnetFC2, dlparams, settings, varargin)
% Check inputs and outputs.
%   Useful assignments:
mode = settings.trainMode;      % Which subnets to train
%   Default input arguments:
nStaticArgs = 8;
nOptArgs    = 2;
%   Default output arguments:
nOptOutputs = 4;
%   Default boolean options:
isCalcGradients = true;
isTrain         = true;
isTrainCNN      = true;
isTrainBN       = true;
%   Argument check:
if (nargin > nStaticArgs + nOptArgs)
    error("Wrong number of input arguments provided");
end
if (nargin > nStaticArgs)
    isCalcGradients = varargin{1};
end
if (nargin > nStaticArgs + 1)
    isTrain = varargin{2};
end
if isempty(find(strcmp(mode, 'cnn')))
    isTrainCNN  = false;
end
if isempty(find(strcmp(mode, 'bn')))
    isTrainBN  = false;
end

% Run model.
if (isTrainCNN && isTrain)
    dlY = squeezenetModel(dlX, dlnetCNN, 'train');
else
    dlY = squeezenetModel(dlX, dlnetCNN, 'test');
end

if (isTrainBN && isTrain)
    dlY = bnModel(dlY);
else
    dlY = bnModel(dlY);
end

dlYPredPos = fcModel(dlY, dlnetFC1);
dlYPredSixdim = fcModel(dlY, dlnetFC2);

% Position loss.
dlYTruePos = dlYTrue(1:3,:);
lossPos = calcPosLoss(dlYPredPos,dlYTruePos);
errPos = double(dlUnprocess(lossPos,settings));

% Attitude loss.
dlYTrueQuat = dlYTrue(4:7,:);
lossAtt = calcAttLoss(dlYPredSixdim,dlYTrueQuat, settings);
errSo3 = calcSo3Error(dlUnprocess(dlYPredSixdim,settings),dlUnprocess(dlYTrueQuat,settings));

% Position/attitude weights.
sp = getDlnetVal(dlparams.Learnables,"sp","Weight");
sq = getDlnetVal(dlparams.Learnables,"sq","Weight");

% Combined loss.
loss = lossPos.*exp(-sp) + sp + lossAtt.*exp(-sq) + sq;

% Compute gradients.
if isCalcGradients
    %  Initialise output variables.
    for i = nOptOutputs
        varargout{i} = {};
    end
    
    nModes = length(mode);
    
    %  Iterate over all chosen modes.
    for i = 1:nModes
        mode_i = mode{i};
        % If not last mode, save gradient tracing for speed.
        if (i ~= nModes)
            retainData = true;
        else
            retainData = false;
        end
        
        switch(mode_i)
            case 'cnn'
                gradCNN = dlgradient(mean(loss), dlnetCNN.Learnables, 'RetainData', retainData);
                varargout{1} = gradCNN;
            case 'bn'
                gradBN = dlgradient(mean(loss), dlnetBN.Learnables, 'RetainData', retainData);
                varargout{2} = gradBN;
             case 'fc1'
                gradFC1 = dlgradient(mean(loss), dlnetFC1.Learnables, 'RetainData', retainData);
                varargout{3} = gradFC1;
            case 'fc2'
                gradFC2 = dlgradient(mean(loss), dlnetFC2.Learnables, 'RetainData', retainData);
                varargout{4} = gradFC2;
            case 'optw'
                gradOptW = dlgradient(mean(loss), dlparams.Learnables, 'RetainData', retainData);
                varargout{5} = gradOptW;
            otherwise
                error('Unsupported module');
        end
        
    end
end
