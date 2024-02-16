function x = leakyrelu(x,scale)
%LEAKYRELU   Apply leaky rectified linear unit activation
%
%   Shamelessly adapted from relu.m

% Extract the input data
fd = x.FormattedData;
x.FormattedData = [];
[fd,xdata] = extractData(fd);

% The dlarray methods should not accept logical x 
if islogical(xdata)
    error(message('deep:dlarray:LogicalsNotSupported'));
end

xdata = max(0,xdata) + scale*min(0,xdata);

% Format is guaranteed not to have changed
fd = insertData(fd, xdata);
x.FormattedData = fd;
end