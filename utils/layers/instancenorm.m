function dlY = instancenorm(dlX)
%   INSTANCENORM
%   Normalise each channel and observation of input data.
%   Does not currently support affine parameters or running stats.
%
%   Inputs:
%       ... - ... - ...

fmt = dlX.dims;
reductionDims = find(fmt == 'S');

mu = mean(dlX,reductionDims);
sigmaSq = var(dlX,1,reductionDims);

epsilon = 1e-5;
dlY = (dlX - mu) ./ sqrt(sigmaSq + epsilon);

end