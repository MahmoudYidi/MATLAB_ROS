function [tree] = ...
        buildConv2d(blockId, settings,tree, kernel, stride, padding, ...
            dimsW, dimsB, varargin)
% buildConv2d
%
% Creates the parameters for to use in a 2D dlarray convolution (dlconv)
% operation.

fan_mode = 'fan_in';
nonlinearity = 'leaky_relu';
nslope = 0.1;

nDefaultArgs = 8;
if nargin > nDefaultArgs
    fan_mode = varargin{1};
end
if nargin > nDefaultArgs+1
    nonlinearity = varargin{2};
end
if nargin > nDefaultArgs+2
    nslope = varargin{3};
end

tree.opLayer = [tree.opLayer; blockId];
tree.opParam = [tree.opParam; "Stride"];
tree.opVal = [tree.opVal; stride];

tree.opLayer = [tree.opLayer; blockId];
tree.opParam = [tree.opParam; "Padding"];
tree.opVal = [tree.opVal; padding];

tree.opLayer = [tree.opLayer; blockId];
tree.opParam = [tree.opParam; "Kernel"];
tree.opVal = [tree.opVal; kernel];

weights = initialiseKaimingNormal(dimsW, 'conv', fan_mode, nonlinearity, nslope);

tree.netLayer = [tree.netLayer; blockId];
tree.netParam = [tree.netParam; "Weights"];
tree.netVal{end+1} = dlProcess(weights,'',settings);

if ~isempty(dimsB)
    bias = zeros(dimsB,'single');
    tree.netLayer = [tree.netLayer; blockId];
    tree.netParam = [tree.netParam; "Bias"];
    tree.netVal{end+1} = dlProcess(bias,'',settings);
end

end

