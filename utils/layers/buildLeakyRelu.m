function [tree] = ...
        buildLeakyRelu(blockId, tree, scale)
% buildLeakyRelu
%
% Creates the parameters for to use in a leaky ReLU operation.

tree.opLayer = [tree.opLayer; blockId];
tree.opParam = [tree.opParam; "Scale"];
tree.opVal = [tree.opVal; scale];

end

