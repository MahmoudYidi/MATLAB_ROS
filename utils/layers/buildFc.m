function [tree] = buildFc(blockId, settings,tree, dimsW, dimsB, varargin)
% buildFc
%
% Creates the parameters for to use in a linear fully connected (FC)
% operation.

fan_mode = 'fan_in';
nonlinearity = 'linear';
nslope = 0;

nDefaultArgs = 5;
if nargin > nDefaultArgs
    fan_mode = varargin{1};
end
if nargin > nDefaultArgs+1
    nonlinearity = varargin{2};
end
if nargin > nDefaultArgs+2
    nslope = varargin{3};
end

weights = initialiseKaimingNormal(dimsW, 'fc', fan_mode, nonlinearity, nslope);
bias = zeros(dimsB,'single');

tree.netLayer = [tree.netLayer; blockId];
tree.netParam = [tree.netParam; "Weights"];
tree.netVal{end+1} = dlProcess(weights,'',settings);

tree.netLayer = [tree.netLayer; blockId];
tree.netParam = [tree.netParam; "Bias"];
tree.netVal{end+1} = dlProcess(bias,'',settings);
end

