function parameter = initialiseOrthogonal(sz)
% initialiseOrthogonal
% The Orthogonal initializer initialise the input weights with Q, the 
% orthogonal matrix given by the QR decomposition of Z = QR for a random
% matrix Z sampled from a unit normal distribution.
%
% Supported types:
%   layer     type                expected input
%   -------------------------------------------------------------------
%   'default' default             [f(out) f(in)]

[parameter,~] = qr(randn(sz,'single'));

end

