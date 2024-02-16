function network = lossParams(settings)
% Learnable weighing factors for pose loss function.
% Follows a probabilistic combination of position and attitude losses based on a Laplace likelihood.
% Source:
%   Kendall & Cipolla (2017): Geometric loss functions for camera pose regression with deep learning

%% Initialise
[network,tmpTree] = layerInitialiser();

%% Weights
sp = 0; % Position
tmpTree.netLayer = [tmpTree.netLayer; "sp"];
tmpTree.netParam = [tmpTree.netParam; "Weight"];
tmpTree.netVal{end+1} = dlProcess(sp,'',settings);

sq = -3;
tmpTree.netLayer = [tmpTree.netLayer; "sq"];
tmpTree.netParam = [tmpTree.netParam; "Weight"];
tmpTree.netVal{end+1} = dlProcess(sq,'',settings);

%% Post-process
network.Learnables = table(tmpTree.netLayer, tmpTree.netParam, tmpTree.netVal', ...
    'VariableNames', ["Layer","Parameter","Value"]);
end

