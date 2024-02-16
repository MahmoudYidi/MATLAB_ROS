function lossValid = calcValidLossStage2RNN(imds, dlnetCNN, dlnetBN, dlnetRNN, dlnetFC, ...
    dlparams, seqBatchSize, settings)
% calcValidLossRNN
%
% Computes the validation loss for one epoch for the RNN.

% Define the desired types of losses
lossLabels = ["TotalLoss";"PositionLoss";"AttitudeLoss"];

nSeqs               = length(imds);
miniBatchSize       = settings.miniBatchSizeValid;
miniBatchSizeCNN    = settings.miniBatchSizeCNN;

doImgAugmentation   = false;
doImgAugConsistency = false;
isCalcGradients     = false;
isStateful          = true;
isTrain             = false;

lossTotalArray  = [];
lossPosArray    = [];
lossAttArray    = [];

% Loop over sequences.
miniBatchGroup = calcGroupByMinibatch(nSeqs,miniBatchSize);
numMiniBatches = height(miniBatchGroup);
for j = 1:numMiniBatches
    lbSeq = miniBatchGroup.Start(j);
    ubSeq = miniBatchGroup.End(j);
    
    % Batch of sequences.
    miniBatch_j = imds(lbSeq:ubSeq);
    miniBatchSize_j = length(miniBatch_j);
    
    % Compute CNN features for the sequences.
    X = cell(miniBatchSize_j, 1);
    Y = cell(miniBatchSize_j, 1);
    for k = 1:miniBatchSize_j
        [X_k, Y_k] = computeCNNFeatures(miniBatch_j{k}, dlnetCNN, dlnetBN, miniBatchSizeCNN, settings, ...
            doImgAugmentation, doImgAugConsistency, 'stage2');
        
        X{k} = X_k;
        Y{k} = Y_k;
    end
    
    % Pre-processing for sequence padding
    maxLength = max(cellfun(@height,miniBatch_j));                  % Maximum sequence length in batch
    maxLengthRound = seqBatchSize*ceil(maxLength/seqBatchSize);     % Round length up to nearest multiple
                                                                    % of seqBatchSize
    
    dlnetRNN.State = eraseRNNState(dlnetRNN.State);
    featuresTrainGroup = calcGroupByMinibatch(maxLengthRound,seqBatchSize,seqBatchSize);
    nSubSeqs = height(featuresTrainGroup);
    for k = 1:nSubSeqs     
        % Build mini-batch.
        % 'CBT'
        [X_k,Y_k] = getSubSequenceBatchPadRight(X, Y, featuresTrainGroup, k, miniBatchSize, seqBatchSize);
        dlX_k = dlProcess(X_k, 'CBT', settings);
        
        % Run RNN.
        [loss, errPos, errAtt, stateRNN, ~, ~] = ...
            dlfeval(@modelGradientsStage2RNN, dlX_k, Y_k, ...
            dlnetRNN, dlnetFC, dlparams, settings, isCalcGradients, isStateful, isTrain);
        if isStateful
            dlnetRNN.State = stateRNN;
        end
        
        % Populate vectors.
        fprintf("j = %d,\t k = %d\n", j, k);
        lossTotalArray  = [lossTotalArray; double(dlUnprocess(loss(:), settings))]; %#ok<*AGROW>
        lossPosArray    = [lossPosArray; errPos(:)];
        lossAttArray    = [lossAttArray; errAtt(:)];
    end
    
end

% Compile into a single table
lossValid = table(lossLabels, [mean(lossTotalArray);mean(lossPosArray);mean(lossAttArray)], ...
    'VariableNames', ["Parameter","Value"]);

end

