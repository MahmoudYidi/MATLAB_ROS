function imdsGroup = calcGroupByMinibatch(nObs,miniBatchSize,varargin)

% Input parsing
nStaticArgs = 2;
stride = miniBatchSize;
if nargin > nStaticArgs
    stride = varargin{1};
end

k = 0;
nIter = ceil((nObs - miniBatchSize)/stride + 1);

colGroup = zeros(nIter,1);
colStart = zeros(nIter,1);
colEnd = zeros(nIter,1);

for j = 1:nIter
    k = k + 1;
    idx = (j - 1)*stride + 1;
    
    diffSz = nObs - idx + 1;
    if (diffSz < miniBatchSize)
        readSz = diffSz - 1;
    else
        readSz = miniBatchSize - 1;
    end
    
    idxGroup = k;
    idxStart = idx;
    idxEnd = idx + readSz;
    
    colGroup(j) = idxGroup;      %#oak<*AGROW>
    colStart(j) = idxStart;
    colEnd(j) = idxEnd;
end

imdsGroup = table(colGroup, colStart, colEnd, ...
    'VariableNames', ["Group","Start","End"]);

end

