function [foundSample, subPartitions, oldPartitions] = getRandomVectorSubset(partitions, prop, tol, maxIts)

subsequences    = partitions.Elements;
numSeqs         = numel(subsequences);
seqLen          = sum(subsequences);

szVal = round(prop*seqLen); % Proportion of frames to sample
foundSample = false;        % Has sucessfully found a sample
subPartitions = [];         % New partition vector with prop elements
oldPartitions = partitions; % Old partitioned vector with subPartitions removed

for count = 1:maxIts
    p = randperm(numSeqs);
    s = cumsum(subsequences(p));
    k = find(abs(s - szVal) < tol);
    if ~isempty(k)
        sample_indices = p(1:k(1));
        foundSample = true;
        break
    end
end

if (foundSample)
    subPartitions = partitions(sample_indices, :);
    oldPartitions(sample_indices, :) = [];
end

end

