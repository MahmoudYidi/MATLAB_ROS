function lossValid = calcValidLoss(imds, dlnetCNN, dlnetBN, dlnetFC1, dlnetFC2, dlparams, settings)
% calcValidLoss
%
% Computes the validation loss for one epoch.

% Pre-processing.
%   Variables:
miniBatchSize   = settings.miniBatchSizeValid;              % Mini-batch size
nObs            = height(imds);                             % No. validation images
group           = calcGroupByMinibatch(nObs,miniBatchSize); % Validation mini-batch group indices
numIter         = height(group);

%   Boolean variables:
doImgAugmentation   = false;                % Perform image augmentation
doImgAugConsistency = false;                % Be consistent in image augmentation
isCalcGradients     = false;                % Compute gradients
isTrain             = false;                % Train
%   Arrays:
lossTotalArray  = [];
lossPosArray    = [];
lossAttArray    = [];
accAtt = 0;

% Define the desired types of losses.
lossLabels = ["TotalLoss";"PositionLoss";"AttitudeLoss";"AttitudeAccuracy"];

for k = 1:numIter
    % Read minibatch.
     [X,Y] = readImdsSeqBatch(imds,group,k,settings,doImgAugmentation,doImgAugConsistency);
     
    % Concatenate and rescale predictors.
    % TODO: move this to preprocessing function.
    X = cat(4, X{:});
    X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
    dlX = dlProcess(X,'SSCB',settings);
    
    % Process responses.
    YTrue = cat(1, Y{:});
    YTrue = YTrue';
    dlY = dlProcess(YTrue,'CB',settings);
    
    [loss, errPos, errAtt, corAtt, ~, ~, ~] = ...
        modelGradientsStage1(dlX, dlY, dlnetCNN, dlnetBN, dlnetFC1, dlnetFC2, dlparams, ...
        settings, isCalcGradients, isTrain);
    
    % Populate vectors.
    lossTotalArray  = [lossTotalArray double(dlUnprocess(loss, settings))]; %#ok<*AGROW>
    lossPosArray    = [lossPosArray errPos];
    lossAttArray    = [lossAttArray errAtt];
    accAtt = accAtt + corAtt;
end

% Average the results
lossTotalValid = mean(lossTotalArray);
lossPosValid = mean(lossPosArray);
lossAttValid = mean(lossAttArray);
accAttValid = accAtt/nObs;

% Compile into a single table
lossValid = table(lossLabels, [lossTotalValid;lossPosValid;lossAttValid;accAttValid], ...
    'VariableNames', ["Parameter","Value"]);

end

