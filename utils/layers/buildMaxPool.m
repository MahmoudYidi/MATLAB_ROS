function [tree] = ...
        buildMaxPool(blockId, tree, kernel, stride, padding)

tree.opLayer = [tree.opLayer; blockId];
tree.opParam = [tree.opParam; "Kernel"];
tree.opVal = [tree.opVal; kernel];

tree.opLayer = [tree.opLayer; blockId];
tree.opParam = [tree.opParam; "Stride"];
tree.opVal = [tree.opVal; stride];

tree.opLayer = [tree.opLayer; blockId];
tree.opParam = [tree.opParam; "Padding"];
tree.opVal = [tree.opVal; padding];

end

