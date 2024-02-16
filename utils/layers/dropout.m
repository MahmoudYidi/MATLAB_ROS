function dlY = dropout(dlX,p)
% Dropout layer

mask = rand(size(dlX));
mask(mask < p) = 0;
mask(mask >= p) = 1/(1-p);

dlY = dlX.*mask;

end