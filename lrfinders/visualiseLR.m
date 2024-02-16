function visualiseLR(settings, miniBatchSize, imdsTrain, dataMode)

switch lower(dataMode)
    case 'random'
        [lrArray,nIter] = visualiseLRRandom(settings, miniBatchSize, imdsTrain);
    case 'sequence'
        [lrArray,nIter] = visualiseLRSequence(settings, miniBatchSize, imdsTrain);
    otherwise
        error('Unknown method');
end

plot(lrArray, 'r','linewidth',1.2);
xlabel("Epoch");
ylabel("Learning rate");
grid off;
grid minor;
box on;

for i = 0:settings.epochLineMult:settings.numEpochs
    vline = xline(i*nIter,':',i,'LabelVerticalAlignment', 'top',...
        'LabelOrientation','horizontal');
    vline.Annotation.LegendInformation.IconDisplayStyle = 'off';
    vline.FontSize = 6;
end

end

function [lrArray,nIter] = visualiseLRSequence(settings,miniBatchSize,imds)

globalIter = 0;
seqIter = 0;
nSeqs = length(imds);
nIter = getNumIterSeq(miniBatchSize, imds, 'cnn');

lrArray = [];

for i = 1:settings.numEpochs
    % Loop over sequences.
    for j = 1:nSeqs
        seqIter = seqIter + 1;
        lr = lrDecayer(i,seqIter,settings,nSeqs);
                    
        nObs = height(imds{j});
        imdsGroup = calcGroupByMinibatch(nObs,miniBatchSize);
        
        nMiniBatchesTrain = height(imdsGroup);
        
        % Loop over mini-batches.
        for k = 1:nMiniBatchesTrain
            globalIter = globalIter + 1;
            lrArray = [lrArray lr]; %#ok<*AGROW>
        end
    end
end

end

function [lrArray,nIter] = visualiseLRRandom(settings,miniBatchSize,imds)

globalIter = 0;

nObs = height(imds);

imdsGroup = calcGroupByMinibatch(nObs,miniBatchSize);
nIter = height(imdsGroup);

lrArray = zeros(nIter*settings.numEpochs,1);

for i = 1:settings.numEpochs
    
    % Loop over mini-batches.
    for k = 1:nIter
        
        globalIter = globalIter + 1;
        lr = lrDecayer(i,globalIter,settings,nIter);
        
        lrArray((i-1)*nIter + k) = lr;
    end
end

end

