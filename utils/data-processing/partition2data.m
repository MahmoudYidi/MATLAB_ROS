function [predictorsPart,responsesPart] = partition2data(partitions,predictors,responses)

nSeqs = height(partitions);
predictorsPart = cell(nSeqs, 1);
responsesPart = cell(nSeqs, 1);

for j = 1:nSeqs
    predictorsPart{j} = predictors(partitions.Start(j):partitions.End(j),:);
    responsesPart{j} = responses(partitions.Start(j):partitions.End(j),:);
end

end

