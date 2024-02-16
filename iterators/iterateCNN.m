function iterateCNN(settings, imdsTrain, imdsValid)

%% Load network parameters
[dlnetCNN,dlnetBN,dlnetFC1,dlnetFC2,dlparams,avgG,avgGS,globalIter,iniEpoch] = loadParams(settings);

%% Loop pre-processing
%  Load some useful variables.
nImgsTrain      = height(imdsTrain);        % No. training images
miniBatchSize   = settings.miniBatchSize;   % Mini-batch size

imdsGroupTrain  = calcGroupByMinibatch(nImgsTrain,miniBatchSize);   % Mini-batch group indices
numIter         = height(imdsGroupTrain);                           % Number of MB iterations

trainMode       = settings.trainMode;    	% Training modes (subnet selection for learning)
nTrainModes     = length(trainMode);        % Number of training modes

doImageAugmentation = settings.doImageAugmentation; % Image augmentation
doImAugConsistency  = false;                        % Sequence consistency

doModPose = true;                                   % Modify pose in image augmentation
doHavePts = false;                                  % Have model points
gtParams = {settings.kMat, doModPose, doHavePts};   % Ground truth parameters for image augmentation

%  Create loss figure.
[lossFig,lossFigLines] = createLossFigCNN(iniEpoch-1,numIter);

%  Start time keeping.
start = tic;

%  Initial validation loss at i = 0.
% lossValid = calcValidLossCNN(imdsValid, dlnetCNN, dlnetBN, dlnetFC1, dlnetFC2, dlparams, settings);
% lossFigLines = plotValidLossCNN(lossValid, lossFigLines, iniEpoch-1, numIter);

%% Training loop
for i = iniEpoch:settings.numEpochs
    
    % Record time at beginning of each epoch.
    startEpoch = tic;
    
    % Shuffle datastores.
    imdsTrainShuffled = shuffleDatastore(imdsTrain);
    
    % Loop over mini-batches.
    for k = 1:numIter
        globalIter = globalIter + 1;
        lr = lrDecayer(i,globalIter,settings,numIter);  
        
        [dataX,dataY] = readImdsSeqBatch(imdsTrainShuffled,imdsGroupTrain,k,settings,...
            doImageAugmentation,doImAugConsistency,{},gtParams);
        
        % Concatenate mini-batch of data.
        X = cat(4, dataX{:});
        Y = cat(1, dataY{:});
        Y = cat(1, Y{:,1});     % Position/quaternion
        Y = Y';
        
        % Normalize the images to [-1 1].
        X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
        
        % Convert mini-batch of data to dlarray specify the dimension
        % labels 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlProcess(X,'SSCB',settings);
        dlY = dlProcess(Y,'CB',settings);
        
        % Evaluate the model gradients using dlfeval.
        [loss, errPos, errAtt, sPos, sAtt, gradCNN, gradBN, gradFC1, gradFC2, gradOptW] = ...
            dlfeval(@modelGradientsCNN, dlX, dlY, dlnetCNN, dlnetBN, dlnetFC1, dlnetFC2, dlparams, settings);
        
        % Update network parameters.
        for kk = 1:nTrainModes
            mode_kk = trainMode{kk};
            
            switch(mode_kk)
                case 'cnn'
                    gradCNN = regulariseL2('cnn',dlnetCNN,gradCNN,settings);
                    
                    [dlnetCNN.Learnables,avgG.CNN,avgGS.CNN] = ...
                        adamupdate(dlnetCNN.Learnables, gradCNN, ...
                        avgG.CNN, avgGS.CNN, globalIter, ...
                        lr, settings.beta1, settings.beta2);             
                case 'bn'
                    gradBN = regulariseL2('bn',dlnetBN,gradBN,settings);
                    
                    [dlnetBN.Learnables,avgG.BN,avgGS.BN] = ...
                        adamupdate(dlnetBN.Learnables, gradBN, ...
                        avgG.BN, avgGS.BN, globalIter, ...
                        lr, settings.beta1, settings.beta2);
                case 'fc1'
                    gradFC1 = regulariseL2('fc1',dlnetFC1,gradFC1,settings);
                    
                    [dlnetFC1.Learnables,avgG.FC1,avgGS.FC1] = ...
                        adamupdate(dlnetFC1.Learnables, gradFC1, ...
                        avgG.FC1, avgGS.FC1, globalIter, ...
                        lr, settings.beta1, settings.beta2);
                case 'fc2'
                    gradFC2 = regulariseL2('fc2',dlnetFC2,gradFC2,settings);
                    
                    [dlnetFC2.Learnables,avgG.FC2,avgGS.FC2] = ...
                        adamupdate(dlnetFC2.Learnables, gradFC2, ...
                        avgG.FC2, avgGS.FC2, globalIter, ...
                        lr, settings.beta1, settings.beta2);
                case 'optw'
                    gradOptW = regulariseL2('optw',dlparams,gradOptW,settings);
                    
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
        
        % Every 'x' iterations, save figure.
        if mod(globalIter,settings.saveFigMult) == 0 || globalIter == 1
            saveas(lossFig,[settings.outFolder '/' 'progress.png']);
            saveas(lossFig,[settings.outFolder '/' 'progress.fig']);
        end

    end
    
    % Print out epoch update.
    elapsedTime = toc(startEpoch);
    disp(['Epoch ' num2str(i) '. '...
        'Time taken for epoch = ' num2str(elapsedTime) 's. '...
        'Learning rate = ' num2str(lr)]);
    
    % Compute and plot validation loss.
    lossValid = calcValidLossCNN(imdsValid, dlnetCNN, dlnetBN, dlnetFC1, dlnetFC2, dlparams, settings);
    lossFigLines = plotValidLossCNN(lossValid, lossFigLines, i, numIter);
    
    if mod(i,settings.epochLineMult) == 0
        vline = xline(lossFigLines{1}.ax,i*numIter,':',i,'LabelVerticalAlignment', 'top',...
            'LabelOrientation','horizontal');
        vline.Annotation.LegendInformation.IconDisplayStyle = 'off';
        vline.FontSize = 6;
        
        save(sprintf('%s/model%d.mat',settings.outFolder,i),'dlnetCNN','dlnetBN','dlnetFC1','dlnetFC2','dlparams',...
            'avgG', 'avgGS','globalIter','lr','i','settings');
    end
    
    % Save figure after each epoch.
    saveas(lossFig,[settings.outFolder '/' 'progress.png']);
    saveas(lossFig,[settings.outFolder '/' 'progress.fig']);
end

end

