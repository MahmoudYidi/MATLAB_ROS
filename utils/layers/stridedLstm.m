function [dlY,H_out,C_out] = stridedLstm(dlX, h0, c0, W, R, B, stride)
%STRIDEDLSTM Summary of this function goes here

% Get temporal dimension.
fmt = dlX.dims;
tDim = find(fmt == 'T');

% Get sequence length.
seqLen = size(dlX, tDim);

if (seqLen == stride)
    [dlY,H_out,C_out] = lstm(dlX,h0,c0,W,R,B);
    
elseif (seqLen > stride)
    
    for i = 1:seqLen    
        dlX_in = getMatSlice(dlX,tDim,i);
        %dlX_in = squeeze(dlX_in);
        
        [dlY_out,H,C] = lstm(dlX_in,h0,c0,W,R,B);
        
        if (i == stride)
           H_out = H;
           C_out = C;
        end
        
        if (i > 1)
            dlY = cat(tDim, dlY, dlY_out);
        else
            dlY = dlY_out;
        end
        h0 = H;
        c0 = C;
    end
    
else
    error("Stride cannot be larger than sequence length");
end

end

