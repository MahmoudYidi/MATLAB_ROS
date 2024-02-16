function network = bnParams()
% Fully connected layer model for bottleneck.

%% Initialise
[network,tmpTreeLearnables] = layerInitialiser();

%% Post-process
network.Learnables = table(tmpTreeLearnables.netLayer, tmpTreeLearnables.netParam, tmpTreeLearnables.netVal', ...
    'VariableNames', ["Layer","Parameter","Value"]);
network.Operators = table(tmpTreeLearnables.opLayer, tmpTreeLearnables.opParam, tmpTreeLearnables.opVal, ...
    'VariableNames', ["Layer","Parameter","Value"]);

end