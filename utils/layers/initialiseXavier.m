function parameter = initialiseXavier(sz, varargin)
% initialiseXavier
% The Xavier (or Glorot) initializer independently samples from a uniform
% distribution with zero mean and variance equal to 2/(numIn + numOut). The
% output will depend on the layer type.
%
% Supported types:
%   layer     type                expected input
%   -------------------------------------------------------------------
%   'conv'    convolution         [s(in) s(in) c(in) f(out)]
%   'fc'      fully connected     [f(out) s(in) s(in) c(in)]
%   'default' default             [f(out) f(in)]

if nargin == 1
    layer = 'default';
else
    layer = varargin{1};
end

switch(layer)
    case 'fc'
        assert(numel(sz) == 4); % TODO: change it
        
        f = sz(1);
        s1 = sz(2);
        s2 = sz(3);
        c = sz(4);
        
        numIn = s1*s2*c;
        numOut = f;
        
    case 'conv'
        assert(numel(sz) == 4);
        
        f = sz(4);
        s1 = sz(1);
        s2 = sz(2);
        c = sz(3);
        
        numIn = s1*s2*c;
        numOut = s1*s2*f;
        
    otherwise
        numIn = sz(1);
        numOut = sz(2);
end

sigma_sq = 2/(numIn + numOut);
parameter = rand(sz,'single').*sqrt(sigma_sq);

end

