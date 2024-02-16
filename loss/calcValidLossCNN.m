function lossValid = calcValidLossCNN(imds, dlnetCNN, dlnetBN, dlnetFC1, dlnetFC2, dlparams, settings)
% calcValidLoss
%
% Computes the validation loss for one epoch.

% Pre-processing.
%   If not a sequence, move imds to a single cell
if ~settings.isSeq
    imds = {imds};
end
%   Variables:
nSeqsTrain      = length(imds);                 % Number of sequences
miniBatchSize   = settings.miniBatchSizeValid;	% Mini-batch size
%   Boolean variables:
doImgAugmentation   = false;                % Perform image augmentation
doImgAugConsistency = false;                % Be consistent in image augmentation
isCalcGradients     = false;                % Compute gradients
isTrain             = false;                % Train
%   Arrays:
lossTotalArray  = [];
lossPosArray    = [];
lossAttArray    = [];

% Define the desired types of losses.
lossLabels = ["TotalLoss";"PositionLoss";"AttitudeLoss"];

% Loop over sequences.
for i = 1:nSeqsTrain
    nObsValid = height(imds{i});                                    % Number of images in Sequence i
    imdsValidGroup = calcGroupByMinibatch(nObsValid,miniBatchSize);	% Number of mini-batches in Sequence i
    
    nMiniBatchesValid = height(imdsValidGroup);
    
    % Loop over mini-batches.
    for j = 1:nMiniBatchesValid
        
        % Read mini-batch of data.
        [dataX,dataY] = readImdsSeqBatch(imds{i},imdsValidGroup,j,settings,doImgAugmentation,doImgAugConsistency);
        
        % Concatenate mini-batch of data.
        X = cat(4, dataX{:});
        Y = cat(1, dataY{:});
        Y = cat(1, Y{:,1});     % Position/quaternion
        Y = Y';
        
        % Normalize the images to [-1 1].
        X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
        
        % Convert mini-batch of data to dlarray specify the dimension
        % labels 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlProcess(X,'SSCB',settings);
        dlY = dlProcess(Y,'CB',settings);
        
        [loss, errPos, errSo3, ~, ~] = ...
            modelGradientsCNN(dlX, dlY, dlnetCNN, dlnetBN, dlnetFC1, dlnetFC2, dlparams, ...
            settings, isCalcGradients, isTrain);
        
        % Populate vectors.
        lossTotalArray  = [lossTotalArray double(dlUnprocess(loss, settings))]; %#ok<*AGROW>
        lossPosArray    = [lossPosArray errPos];
        lossAttArray    = [lossAttArray errSo3];
    end
end

% Average the results
lossTotalValid = mean(lossTotalArray);
lossPosValid = mean(lossPosArray);
lossAttValid = mean(lossAttArray);

% Compile into a single table
lossValid = table(lossLabels, [lossTotalValid;lossPosValid;lossAttValid], ...
    'VariableNames', ["Parameter","Value"]);

end

