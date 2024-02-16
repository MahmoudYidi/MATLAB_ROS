function [network,tmpTree] = layerInitialiser()
% layerInitialiser
network = struct;       % Holds output
tmpTree = struct;       % Holds temporary cells that will be parsed into
                        % output

tmpTree.netLayer = cell(0);     % Holds learnables
tmpTree.netParam = cell(0);
tmpTree.netVal = cell(0);

tmpTree.opLayer = cell(0);      % Holds operator sizes, etc.
tmpTree.opParam = cell(0);
tmpTree.opVal = cell(0);

end

