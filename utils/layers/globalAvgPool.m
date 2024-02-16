function dlY = globalAvgPool(dlX)
%   INSTANCENORM
%   Normalise each channel and observation of input data.
%   Does not currently support affine parameters or running stats.
%
%   Inputs:
%       ... - ... - ...

fmt = dlX.dims;
reductionDims = find(fmt == 'S');

dlY = mean(dlX,reductionDims);

end