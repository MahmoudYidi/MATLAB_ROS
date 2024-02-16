function [loss, errPos, errSo3, state, sp, sq, varargout] = ...
    modelGradientsStage2RNN(dlX, YTrue, dlnetRNN, dlnetFC, dlparams, settings, varargin)
% modelGradientsStage2RNN
% Assumes right-padded sequences

%   Useful assignments:
mode    = settings.trainMode;	% Which subnets to train
stride  = settings.seqStride;	% Temporal sequence stride
%   Default input arguments:
nStaticArgs = 6;
nOptArgs    = 3;
%   Default output arguments:
nOptOutputs = 3;
%   Default boolean options:
isCalcGradients = true;
isStateful      = true;
isTrain         = true;
isTrainRNN      = true;
%   Argument check:
if (nargin > nStaticArgs + nOptArgs)
    error("Wrong number of input arguments provided");
end
if (nargin > nStaticArgs)
    isCalcGradients = varargin{1};
end
if (nargin > nStaticArgs + 1)
    isStateful = varargin{2};
end
if (nargin > nStaticArgs + 2)
    isTrain = varargin{3};
end
if isempty(find(strcmp(mode, 'rnn')))
    isTrainRNN  = false;
end
if (isTrainRNN && isTrain)
    modeRNN = 'train';
else
    modeRNN = 'test';
end

% Run model.
if isStateful
    [dlY,state] = rnnModel(dlX, dlnetRNN, stride, 'stateful', modeRNN);
else
    dlY = rnnModel(dlX, dlnetRNN, stride, 'stateless', modeRNN);
    state = {};
end

dlY = fcModel(dlY, dlnetFC);

% Loss computation.
% Per-batch basis
nBatches    = length(YTrue);                                % Number of batches
lossPos  	= dlProcess(zeros(nBatches,1), 'B', settings);	% Position loss
lossAtt     = dlProcess(zeros(nBatches,1), 'B', settings);	% Attitude loss
errPos      = zeros(nBatches,1);                          	% Position error
errSo3      = zeros(nBatches,1);                          	% SO(3) error

hasEmptyBatch = false;

for i = 1:nBatches
    YTrue_i = YTrue{i};            	% True poses for batch i
    nTimeSteps = size(YTrue_i, 2);	% Number of timesteps
    
    if nTimeSteps == 0
        hasEmptyBatch = true;
        lossPos(i) = nan;
        errPos(i) = nan;
        lossAtt(i) = nan;
        errSo3(i) = nan;
        continue;
    end
    
    % Position loss.
    YTruePos = YTrue_i(1:3,:);      % True position
    dlYTruePos = dlProcess(YTruePos, 'CT', settings);
    
    dlYPredPos = dlY(1:3,i,1:nTimeSteps); % Predicted position
    dlYPredPos = squeeze(dlYPredPos);
    
    lossPos_i = calcPosLoss(dlYPredPos,dlYTruePos);
    lossPos(i) = mean(lossPos_i);
    
    errPos_i = double(dlUnprocess(lossPos(i),settings));
    errPos(i) = errPos_i;
    
    % Attitude loss.
    YTrueQuat = YTrue_i(4:7,:);             % True quaternion
    dlYTrueQuat = dlProcess(YTrueQuat, 'CT', settings);
    
    dlYPredSixdim = dlY(4:9,i,1:nTimeSteps);  % Predicted sixdim
    dlYPredSixdim = squeeze(dlYPredSixdim);
    
    lossAtt_i = calcAttLoss(dlYPredSixdim,dlYTrueQuat, settings);
    lossAtt(i) = mean(lossAtt_i);
    
    errSo3_i = calcSo3Error(dlUnprocess(dlYPredSixdim,settings),YTrueQuat);
    errSo3(i) = mean(errSo3_i);  
end

if (hasEmptyBatch)
    validIdx = ~isnan(extractdata(lossPos));
    
    lossPos = lossPos(validIdx);
    errPos = errPos(validIdx);
    lossAtt = lossAtt(validIdx);
    errSo3 = errSo3(validIdx);
    
    assert(~isempty(lossPos));
end

% Position/attitude weights.
sp = getDlnetVal(dlparams.Learnables,"sp","Weight");
sq = getDlnetVal(dlparams.Learnables,"sq","Weight");

% Compute combined loss.
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
            case 'rnn'
                gradRNN = dlgradient(mean(loss), dlnetRNN.Learnables, 'RetainData', retainData);
                varargout{1} = gradRNN;
            case 'fc'
                gradFC = dlgradient(mean(loss), dlnetFC.Learnables, 'RetainData', retainData);
                varargout{2} = gradFC;
            case 'optw'
                gradOptW = dlgradient(mean(loss), dlparams.Learnables, 'RetainData', retainData);
                varargout{3} = gradOptW;
            otherwise
                error('Unsupported module');
        end
        
    end
end

end
