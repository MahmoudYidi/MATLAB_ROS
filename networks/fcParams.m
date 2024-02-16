function network = fcParams(hyperparameters, nNodesIn, nNodesOut, applySoftmax)
% Fully connected layer model.
% To be used in last step (classification or regression).

%% Initialise
[network,tmpTreeLearnables] = layerInitialiser();

%% Fully connected
tmpTreeLearnables = buildFc("fc", hyperparameters,tmpTreeLearnables,...
                            [nNodesOut,1,1,nNodesIn], [nNodesOut,1,1,1]);

tmpTreeLearnables.opLayer = [tmpTreeLearnables.opLayer; "fc"];
tmpTreeLearnables.opParam = [tmpTreeLearnables.opParam; "Softmax"];
tmpTreeLearnables.opVal{end+1} = applySoftmax;
                        
%% Post-process
network.Learnables = table(tmpTreeLearnables.netLayer, tmpTreeLearnables.netParam, tmpTreeLearnables.netVal', ...
    'VariableNames', ["Layer","Parameter","Value"]);
network.Operators = table(tmpTreeLearnables.opLayer, tmpTreeLearnables.opParam, tmpTreeLearnables.opVal, ...
    'VariableNames', ["Layer","Parameter","Value"]);
end

