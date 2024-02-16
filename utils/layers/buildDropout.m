function [tree] = ...
        buildDropout(blockId, tree, rate)

tree.opLayer = [tree.opLayer; blockId];
tree.opParam = [tree.opParam; "Rate"];
tree.opVal = [tree.opVal; rate];

end

