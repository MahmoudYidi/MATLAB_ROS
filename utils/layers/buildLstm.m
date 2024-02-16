function [learnTree,stateTree] = ...
        buildLstm(blockId, settings,learnTree, stateTree, numHiddenUnits, numFeatures)
%
% buildLstm
%
% Creates the parameters for to use in a LSTM layer.

stateTree.netLayer = [stateTree.netLayer; blockId];
stateTree.netParam = [stateTree.netParam; "HiddenState"];
stateTree.netVal = [stateTree.netVal; zeros(numHiddenUnits,1)];

stateTree.netLayer = [stateTree.netLayer; blockId];
stateTree.netParam = [stateTree.netParam; "CellState"];
stateTree.netVal = [stateTree.netVal; zeros(numHiddenUnits,1)];

% Weights
weights = zeros(4*numHiddenUnits,numFeatures);
recurrentWeights = zeros(4*numHiddenUnits,numHiddenUnits);
for i = 1:4
    l1 = (i-1)*numHiddenUnits + 1;
    l2 = l1 + numHiddenUnits - 1;
    % Standard
    weights(l1:l2,:) = initialiseXavier([numHiddenUnits,numFeatures],'default');
    % Recurrent
    recurrentWeights(l1:l2,:) = initialiseOrthogonal([numHiddenUnits,numHiddenUnits]);
end

% Bias
% From https://uk.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.lstmlayer.html:
% The bias vector is a concatenation of the four bias vectors for the components (gates) 
% in the LSTM layer. The four vectors are concatenated vertically in the following order:
% 1. Input gate
% 2. Forget gate
% 3. Cell candidate
% 4. Output gate
bias = vertcat(...
    zeros(numHiddenUnits,1,'single'), ...
    ones(numHiddenUnits,1,'single'), ... %ones
    zeros(numHiddenUnits,1,'single'), ...
    zeros(numHiddenUnits,1,'single'));

learnTree.netLayer = [learnTree.netLayer; blockId];
learnTree.netParam = [learnTree.netParam; "Weights"];
learnTree.netVal{end+1} = dlProcess(weights,'CU',settings);

learnTree.netLayer = [learnTree.netLayer; blockId];
learnTree.netParam = [learnTree.netParam; "RecurrentWeights"];
learnTree.netVal{end+1} = dlProcess(recurrentWeights,'CU',settings);

learnTree.netLayer = [learnTree.netLayer; blockId];
learnTree.netParam = [learnTree.netParam; "Bias"];
learnTree.netVal{end+1} = dlProcess(bias,'C',settings);

learnTree.netLayer = [learnTree.netLayer; blockId];
learnTree.netParam = [learnTree.netParam; "Alpha1"];
learnTree.netVal{end+1} = dlProcess(ones(4*numHiddenUnits,1),'C',settings);

learnTree.netLayer = [learnTree.netLayer; blockId];
learnTree.netParam = [learnTree.netParam; "Alpha2"];
learnTree.netVal{end+1} = dlProcess(ones(4*numHiddenUnits,1),'C',settings);

learnTree.netLayer = [learnTree.netLayer; blockId];
learnTree.netParam = [learnTree.netParam; "Alpha3"];
learnTree.netVal{end+1} = dlProcess(ones(numHiddenUnits,1),'C',settings);

learnTree.netLayer = [learnTree.netLayer; blockId];
learnTree.netParam = [learnTree.netParam; "Beta3"];
learnTree.netVal{end+1} = dlProcess(zeros(numHiddenUnits,1),'C',settings);
end

