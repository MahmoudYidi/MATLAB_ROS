function [imdsSeqsShuffled] = shuffleDatastore(imdsSeqs)

if istable(imdsSeqs)
    numSeqs = height(imdsSeqs);
else
    numSeqs = length(imdsSeqs);
end

idx = randperm(numSeqs);

imdsSeqsShuffled = imdsSeqs(idx,:);

end

