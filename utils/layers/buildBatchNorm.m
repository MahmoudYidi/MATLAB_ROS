function [learnTree, stateTree] = ...
        buildBatchNorm(blockId, settings, learnTree, stateTree, channels)
% buildLeakyRelu
%
% Creates the parameters for to use in a batch normalisation operation.

% Learnables.
offset = zeros(1,channels);
learnTree.netLayer = [learnTree.netLayer; blockId];
learnTree.netParam = [learnTree.netParam; "Offset"];
learnTree.netVal{end+1} = dlProcess(offset,'',settings);

scale = ones(1,channels);
learnTree.netLayer = [learnTree.netLayer; blockId];
learnTree.netParam = [learnTree.netParam; "Scale"];
learnTree.netVal{end+1} = dlProcess(scale,'',settings);

% State.
mean = zeros(1,channels);
stateTree.netLayer = [stateTree.netLayer; blockId];
stateTree.netParam = [stateTree.netParam; "TrainedMean"];
stateTree.netVal{end+1} = single(mean);

variance = ones(1,channels);
stateTree.netLayer = [stateTree.netLayer; blockId];
stateTree.netParam = [stateTree.netParam; "TrainedVariance"];
stateTree.netVal{end+1} = single(variance);

end

