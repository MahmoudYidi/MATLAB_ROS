function iterateStage2RNN(settings, imdsTrain, imdsValid)

%% Load network parameters
[dlnetCNN,dlnetBN,dlnetRNN,dlnetFC,dlparams,avgG,avgGS,globalIter,iniEpoch,seqIter] = ...
    loadParamsRNN(settings);

%% Loop pre-processing
%  Load some useful variables.
%  Batch sizes:
miniBatchSize       = settings.miniBatchSize;       % Mini-batch size
seqBatchSize        = settings.seqBatchSize;        % Sequence size for RNN
miniBatchSizeCNN    = settings.miniBatchSizeCNN;    % Batch size for feature computation
seqBatchSizeValid   = settings.seqBatchSizeValid;   % Batch size for truncated validation
%  Sequences:
nSeqsTrain  = length(imdsTrain);    % Number of training sequences
seqStride   = settings.seqStride;   % Sequence stride size
stateful    = settings.stateful;    % Stateful training bool
%  Iteration:
numIter             = settings.nIter;                                   % Number of training iterations
miniBatchTrainGroup = calcGroupByMinibatch(nSeqsTrain,miniBatchSize);   % Mini-batch index groups
numMiniBatches      = height(miniBatchTrainGroup);                      % Number of mini-batches
%  Image augmentation:
doImageAugmentation = settings.doImageAugmentation;
doImAugConsistency  = true;     % Sequence consistency
%  Training modes:
trainMode       = settings.trainMode;	% Training modes (subnet selection for learning)
nTrainModes     = length(trainMode);    % Number of training modes
%  GT params:
doModPose = true;                                   % Modify pose in image augmentation
doHavePts = true;                                   % Have model points
gtParams = {settings.kMat, doModPose, doHavePts};   % Ground truth parameters for image augmentation

%  Create loss figure.
[lossFig,lossFigLines] = createLossFigStage2(iniEpoch-1,numIter);

%  Start time keeping.
start = tic;

%  Initial validation loss at i = 0.
% lossValid = calcValidLossStage2RNN(imdsValid, dlnetCNN, dlnetBN, dlnetRNN, dlnetFC, dlparams, ...
%      seqBatchSizeValid, settings);
% lossFigLines = plotValidLossStage2(lossValid, lossFigLines, iniEpoch-1, numIter);

%% Training loop
for i = iniEpoch:settings.numEpochs
    
    % Record time at beginning of each epoch.
    startEpoch = tic;
    
    % Shuffle datastores.
    imdsTrainShuffled = shuffleDatastore(imdsTrain);
    
    % Loop over sequences.
    for j = 1:height(miniBatchTrainGroup)
        % Update seq. no. and learning rate (if applicable).
        seqIter = seqIter + 1;
        lr = lrDecayer(i,seqIter,settings,numMiniBatches);
        
        lbSeq = miniBatchTrainGroup.Start(j);
        ubSeq = miniBatchTrainGroup.End(j);
        
        % Batch of sequences.
        miniBatch_j = imdsTrainShuffled(lbSeq:ubSeq);
        miniBatchSize_j = length(miniBatch_j);

        % Compute CNN features for the sequences.
        X = cell(miniBatchSize_j, 1);
        Y = cell(miniBatchSize_j, 1);
        for k = 1:miniBatchSize_j
            [X_k, Y_k] = computeCNNFeatures(miniBatch_j{k}, dlnetCNN, dlnetBN, miniBatchSizeCNN, settings, ...
                doImageAugmentation, doImAugConsistency, gtParams, 'stage2');
            
            X{k} = X_k;
            Y{k} = Y_k;
        end
        
        % Pre-processing for sequence padding
        maxLength = max(cellfun(@height,miniBatch_j));                  % Maximum sequence length in batch
        maxLengthRound = seqBatchSize*ceil(maxLength/seqBatchSize);     % Round length up to nearest multiple
                                                                        % of seqBatchSize
        
        % Erase RNN state before each sequence.
        if stateful
            dlnetRNN.State = eraseRNNState(dlnetRNN.State, 'gaussian');
        end
        
        featuresTrainGroup = calcGroupByMinibatch(maxLengthRound,seqBatchSize,seqStride);
        nSubSeqs = height(featuresTrainGroup);
        for k = 1:nSubSeqs
            % Update it. no.
            globalIter = globalIter + 1;
            
            % Build mini-batch.
            % 'CBT'
            [X_k,Y_k] = getSubSequenceBatchPadRight(X, Y, featuresTrainGroup, k, miniBatchSize, seqBatchSize);
            dlX_k = dlProcess(X_k, 'CBT', settings);
            
            % Train RNN.
            [loss, errPos, errAtt, stateRNN, sPos, sAtt, gradRNN, gradFC, gradOptW] = ...
                dlfeval(@modelGradientsStage2RNN, dlX_k, Y_k, ...
                dlnetRNN, dlnetFC, dlparams, settings, true, stateful, true);
            if stateful
                dlnetRNN.State = stateRNN;
            end
            
            % Clip gradients.
            switch settings.gradThreshMethod
                case "global-l2norm"
                    gradRNN = thresholdGlobalL2Norm(gradRNN, settings.gradThresh);
                    gradFC = thresholdGlobalL2Norm(gradFC, settings.gradThresh);
                    gradOptW = thresholdGlobalL2Norm(gradOptW, settings.gradThresh);
                case "l2norm"
                    gradRNN = dlupdate(@(g) thresholdL2Norm(g, settings.gradThresh), gradRNN);
                    gradFC = dlupdate(@(g) thresholdL2Norm(g, settings.gradThresh), gradFC);
                    gradOptW = dlupdate(@(g) thresholdL2Norm(g, settings.gradThresh), gradOptW);
                case "absolute-value"
                    gradRNN = dlupdate(@(g) thresholdAbsoluteValue(g, settings.gradThresh), gradRNN);
                    gradFC = dlupdate(@(g) thresholdAbsoluteValue(g, settings.gradThresh), gradFC);
                    gradOptW = dlupdate(@(g) thresholdAbsoluteValue(g, settings.gradThresh), gradOptW);
            end

            % Update network parameters.
            for kk = 1:nTrainModes
                mode_kk = trainMode{kk};
                
                switch(mode_kk)
                    case 'rnn'
                        [dlnetRNN.Learnables,avgG.RNN,avgGS.RNN] = ...
                            adamupdate(dlnetRNN.Learnables, gradRNN, ...
                            avgG.RNN, avgGS.RNN, globalIter, ...
                            lr, settings.beta1, settings.beta2);
                    case 'fc'
                        [dlnetFC.Learnables,avgG.FC,avgGS.FC] = ...
                            adamupdate(dlnetFC.Learnables, gradFC, ...
                            avgG.FC, avgGS.FC, globalIter, ...
                            lr, settings.beta1, settings.beta2);
                    case 'optw'
                        [dlparams.Learnables,avgG.OptW,avgGS.OptW] = ...
                            adamupdate(dlparams.Learnables, gradOptW, ...
                            avgG.OptW, avgGS.OptW, globalIter, ...
                            lr, settings.beta1, settings.beta2);
                end
            end
                        
            % Plot training progress and compute auxiliary figures.
            % Total loss
            lossTrain = mean(double(dlUnprocess(loss, settings)));
            addpoints(lossFigLines{1}.loss.train,globalIter,lossTrain);
            addpoints(lossFigLines{2}.loss.train,globalIter,lossTrain);
            
            % Position loss
            lossPos = mean(errPos);
            addpoints(lossFigLines{3}.loss.train,globalIter,lossPos);
            
            % Attitude loss
            lossAtt = mean(errAtt);
            addpoints(lossFigLines{4}.loss.train,globalIter,lossAtt);
            
            % Learning rate
            addpoints(lossFigLines{5}.lr,globalIter,lr);
            
            % Weights
            sp = double(dlUnprocess(sPos, settings));
            sq = double(dlUnprocess(sAtt, settings));
            addpoints(lossFigLines{6}.sp,globalIter,sp);
            addpoints(lossFigLines{6}.sq,globalIter,sq);
            
            % Elapsed time
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            sgtitle(lossFig,"Epoch: " + i + ", Elapsed: " + string(D));
            drawnow
            
            % Every "x" iterations, save figure.
            if mod(globalIter,settings.saveFigMult) == 0 || globalIter == 1
                saveas(lossFig,[settings.outFolder '/' 'progress.png']);
                saveas(lossFig,[settings.outFolder '/' 'progress.fig']);
            end
        end
    end
    
    % Print out epoch update.
    elapsedTime = toc(startEpoch);
    disp(['Epoch ' num2str(i) '. '...
        'Time taken for epoch = ' num2str(elapsedTime) 's. '...
        'Learning rate = ' num2str(lr)]);
    
    % Compute and plot validation loss.
    lossValid = calcValidLossStage2RNN(imdsValid, dlnetCNN, dlnetBN, dlnetRNN, dlnetFC, dlparams,...
        seqBatchSizeValid, settings);
    lossFigLines = plotValidLossStage1(lossValid, lossFigLines, i, numIter);
    
    if mod(i,settings.epochLineMult) == 0
        vline = xline(lossFigLines{1}.ax,i*numIter,':',i,'LabelVerticalAlignment', 'top',...
            'LabelOrientation','horizontal');
        vline.Annotation.LegendInformation.IconDisplayStyle = 'off';
        vline.FontSize = 6;
        
        save(sprintf('%s/model%d.mat',settings.outFolder,i),'dlnetRNN','dlnetFC','dlparams',...
            'avgG', 'avgGS', 'seqIter','globalIter','lr','i','settings');
    end
    
    % Save figure after each epoch.
    saveas(lossFig,[settings.outFolder '/' 'progress.png']);
    saveas(lossFig,[settings.outFolder '/' 'progress.fig']);
end

end

