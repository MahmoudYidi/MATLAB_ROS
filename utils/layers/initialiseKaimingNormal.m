function parameter = initialiseKaimingNormal(sz, type, mode, nonlinearity, varargin)
% initialiseKaimingNormal
%
% Supported types:
%   layer     type                expected input
%   -------------------------------------------------------------------
%   'conv'    convolution         [s(in) s(in) c(in) f(out)]
%   'fc'      fully connected     [f(out) s(in) s(in) c(in)]

switch(type)
    case 'conv'
        tensor = sz;
        
    case 'fc'
        tensor = [sz(1) sz(4)];
        
    otherwise
        error('Invalid layer');
end

[fan_in, fan_out] = calculateFanInFanOut(tensor);
    
switch(mode)
    case 'fan_out'
        fan = fan_out;
        
    otherwise
        fan = fan_in;
end

switch(nonlinearity)
    case 'leaky_relu'
        if nargin == 4
            nslope = 0.01;
        else
            nslope = varargin{1};
        end
        
        gain = sqrt(2/(1 + nslope^2));
        
    case 'relu'
        gain = sqrt(2);
        
    case 'linear'
        gain = 1;
        
    otherwise
        error('Unsupported nonlinearity');
end

std = gain/sqrt(fan);
parameter = rand(sz,'single').*std^2;

end
