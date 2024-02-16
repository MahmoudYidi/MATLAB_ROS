function [sampler, combinedIdx] = buildStage1Sampler(attBins, posBins, imds)

%  Attitude bin x Position bin x Number of observations x Filename list.
numAttClasses = numel(attBins);
numPosClasses = numel(posBins) - 1;
attIdx = cell(numAttClasses, 1);
posIdx = cell(numPosClasses, 1);
sampler = cell(numAttClasses, numPosClasses);
combinedIdx = zeros(height(imds), 1);

%  Get the indices for each class.
for i = 1:numAttClasses
    attIdx{i} = find(imds.Attitude_Class == i);
end
for i = 1:numPosClasses
    posIdx{i} = find(imds.Position_Class == i);
end

%  Bin.
k = 0;
for i = 1:numAttClasses
   % Get indices for attitude class i
   idx_att = attIdx{i};
   
   for j = 1:numPosClasses
       % Get indices for position class j
       idx_pos = posIdx{j};
       
       idx_pose = intersect(idx_att, idx_pos);
       
       if (~isempty(idx_pose))
           k = k + 1;
           combinedIdx(idx_pose) = k;
       end
       
       % Store in sampler.
       sampler{i,j} = imds(idx_pose,:);
   end
end

end

