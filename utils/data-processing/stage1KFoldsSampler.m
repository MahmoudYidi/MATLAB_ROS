function imdsSampled = stage1KFoldsSampler(sampler, numImgsPerClass)

% classesAll = imds.Pose_Class;
% classesEdges = unique(classesAll);
% classesCounts = histc(classesAll, classesEdges);
% 
% nClasses = numel(classesEdges);
% maxNImgs = max(classesCounts);

switch num2str(numImgsPerClass)
    case '-1'
        imdsSampled = vertcat(sampler{:});
        return;
        
    case '0'
        nSamplesTgt = max(sum(cellfun(@(x) height(x), sampler),2));
        
    otherwise
        nSamplesTgt = numImgsPerClass;
        
end

[nClassesAtt, nClassesPos] = size(sampler);

imdsSampled = table();

for i = 1:nClassesAtt
    nObsArray = cellfun(@(x) height(x), sampler(i,:));
    idxNonZero = find(nObsArray > 0);
    totalNonZero = length(idxNonZero);
    
    if (totalNonZero == 0)
        continue;
    end
    
    splitSamples = divideSamples(nSamplesTgt, totalNonZero);
    
    for j = 1:totalNonZero
        idx_ij = idxNonZero(j);
        nSamples_ij = nObsArray(idx_ij);
        nSamplesTgt_ij = splitSamples(j);
        
        if (nSamples_ij <= nSamplesTgt_ij)
            imdsSampled = vertcat(imdsSampled, sampler{i, idx_ij});
            diffSamples = nSamplesTgt_ij - nSamples_ij;
        else
            diffSamples = nSamplesTgt_ij;
        end
        
        sampleIdxs = randi(nObsArray(idx_ij), diffSamples, 1);
        
        imdsSampled = vertcat(imdsSampled, sampler{i, idx_ij}(sampleIdxs,:));
    end
end

% for i = 1:nClasses
%     class_i = classesEdges(i);
%     classIdxs_i = find(classesAll == class_i);
%     nSamples_i = numel(classIdxs_i);
% 
%     if (numImgsPerClass == 0)
%         imdsSampled = vertcat(imdsSampled, imds(classIdxs_i,:));
%         nSamplesTgt_i = maxNImgs;
%     elseif (numImgsPerClass >= 0)
%         nSamplesTgt_i = numImgsPerClass;
%         if (nSamples_i <= nSamplesTgt_i)
%             imdsSampled = vertcat(imdsSampled, imds(classIdxs_i,:));
%         end
%     else
%         error('Method not yet implemented');
%     end
% 
%     if (nSamples_i == nSamplesTgt_i)
%         continue;
%     end
%     
%     diffSamples = nSamplesTgt_i - nSamples_i;
% 
%     sampleIdxs = classIdxs_i(randi(nSamples_i, diffSamples, 1));
%     
%     imdsSampled = vertcat(imdsSampled, imds(sampleIdxs,:));
end

function splitSamples = divideSamples(totalSamples, N)

    splitSamples = floor(totalSamples ./ repmat(N,1,N));
    remainder = totalSamples - sum(splitSamples);
    splitSamples = splitSamples + [ones(1,remainder) zeros(1,N - remainder)];
    
end

