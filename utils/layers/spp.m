function dlY = spp(dlX,grid)
% Spatial pyramid pooling operation
% Adapted from https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py

fmt = dlX.dims;
sDims = find(fmt == 'S');
bDim = find(fmt == 'B');
cDim = find(fmt == 'C');

c_pconv = size(dlX, cDim);
h_pconv = size(dlX, sDims(1));
w_pconv = size(dlX, sDims(2));
b_pconv = size(dlX, bDim);

dlY = [];

for i = 1:length(grid)
    h_wid = ceil(h_pconv/grid(i));
    w_wid = ceil(w_pconv/grid(i));
    
    h_pad = floor((h_wid*grid(i) - h_pconv + 1)/2);
    w_pad = floor((w_wid*grid(i) - w_pconv + 1)/2);
    
    pooled = maxpool(dlX, [h_wid,w_wid], 'Stride', [h_wid,w_wid], 'Padding', [h_pad,w_pad]);
    
    pooled = stripdims(pooled);
    pooled = permute(pooled, [cDim sDims bDim]);
    pooled = reshape(pooled, [c_pconv*grid(i)*grid(i) b_pconv]);
    
    dlY = cat(1,dlY,pooled);
end

dlY = dlarray(dlY, 'CB');

end