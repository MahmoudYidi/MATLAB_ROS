function [tree] = ...
        buildSpp(blockId, tree, grid)
% buildSpp
%
% Creates the parameters for to use in a spatial pooling pyramid operation.
% Sources:
%   https://blog.acolyer.org/2017/03/21/convolution-neural-nets-part-2/
%   https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py

tree.opLayer = [tree.opLayer; blockId];
tree.opParam = [tree.opParam; "Grid"];
tree.opVal = [tree.opVal; grid];

end

