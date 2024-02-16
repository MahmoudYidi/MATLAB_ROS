function findLRStage2CNN(settings, imdsTrain)

%% Load network parameters
[dlnetCNN,dlnetBN,dlnetFC1,dlnetFC2,dlparams,avgG,avgGS,~,~] = loadParamsStage2(settings);

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
doHavePts = true;                                   % Have model points
gtParams = {settings.kMat, doModPose, doHavePts};   % Ground truth parameters for image augmentation

%  Create LR sweep objects.
lrArray	= logspace(log10(settings.lr_finder.minLr), log10(settings.lr_finder.maxLr), ...
    numIter*settings.lr_finder.epochs);
[lossFig,lossFigLine] = createLRFinderFigStage1(settings);
avgLoss     = 0;
beta        = 0.98;
globalIter  = 0;

%  Start time keeping.
start = tic;

%% Training loop
for i = 1:settings.lr_finder.epochs
    
    % Record time at beginning of each epoch.
    startEpoch = tic;
    
    % Shuffle datastores.
    imdsTrainShuffled = shuffleDatastore(imdsTrain);
    
    % Loop over mini-batches.
    for k = 1:numIter
        % Update it. no. and learning rate (if applicable).
        globalIter = globalIter + 1;
        lr = lrArray(globalIter);
        
        % Read mini-batch of data.
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
        [loss, ~, ~, state, ~, ~, gradCNN, gradBN, gradFC1, gradFC2, gradOptW] = ...
            dlfeval(@modelGradientsStage2CNN, dlX, dlY, dlnetCNN, dlnetBN, dlnetFC1, dlnetFC2, dlparams, settings);
        if ~isempty(find(strcmp(trainMode, 'cnn')))
            dlnetCNN.State = state;
        end
        
        % Update network parameters.
        for kk = 1:nTrainModes
            mode_kk = trainMode{kk};
            
            switch(mode_kk)
                case 'cnn'
                    if (settings.l2Reg.cnn.do == true)
                        l2RegLambda = settings.l2Reg.cnn.lambda;
                        idx = dlnetCNN.Learnables.Parameter == "Weights";
                        gradCNN(idx,:) = dlupdate(@(g,w) g + l2RegLambda*w, ...
                            gradCNN(idx,:), dlnetCNN.Learnables(idx,:));
                    end
                    
                    [dlnetCNN.Learnables,avgG.CNN,avgGS.CNN] = ...
                        adamupdate(dlnetCNN.Learnables, gradCNN, ...
                        avgG.CNN, avgGS.CNN, globalIter, ...
                        lr, settings.beta1, settings.beta2);
                case 'bn'
                    if (settings.l2Reg.bn.do == true)
                        l2RegLambda = settings.l2Reg.bn.lambda;
                        idx = dlnetBN.Learnables.Parameter == "Weights";
                        gradBN(idx,:) = dlupdate(@(g,w) g + l2RegLambda*w, ...
                            gradBN(idx,:), dlnetBN.Learnables(idx,:));
                    end
                    
                    [dlnetBN.Learnables,avgG.BN,avgGS.BN] = ...
                        adamupdate(dlnetBN.Learnables, gradBN, ...
                        avgG.BN, avgGS.BN, globalIter, ...
                        lr, settings.beta1, settings.beta2);
                case 'fc1'
                    if (settings.l2Reg.fc1.do == true)
                        l2RegLambda = settings.l2Reg.fc1.lambda;
                        idx = dlnetFC1.Learnables.Parameter == "Weights";
                        gradFC1(idx,:) = dlupdate(@(g,w) g + l2RegLambda*w, ...
                            gradFC1(idx,:), dlnetFC1.Learnables(idx,:));
                    end
                    
                    [dlnetFC1.Learnables,avgG.FC1,avgGS.FC1] = ...
                        adamupdate(dlnetFC1.Learnables, gradFC1, ...
                        avgG.FC1, avgGS.FC1, globalIter, ...
                        lr, settings.beta1, settings.beta2);
                case 'fc2'
                    if (settings.l2Reg.fc2.do == true)
                        l2RegLambda = settings.l2Reg.fc2.lambda;
                        idx = dlnetFC2.Learnables.Parameter == "Weights";
                        gradFC2(idx,:) = dlupdate(@(g,w) g + l2RegLambda*w, ...
                            gradFC2(idx,:), dlnetFC2.Learnables(idx,:));
                    end
                    
                    [dlnetFC2.Learnables,avgG.FC2,avgGS.FC2] = ...
                        adamupdate(dlnetFC2.Learnables, gradFC2, ...
                        avgG.FC2, avgGS.FC2, globalIter, ...
                        lr, settings.beta1, settings.beta2);
                case 'optw'
                    if (settings.l2Reg.optw.do == true)
                        l2RegLambda = settings.l2Reg.optw.lambda;
                        idx = dlparams.Learnables.Parameter == "Weights";
                        gradOptW(idx,:) = dlupdate(@(g,w) g + l2RegLambda*w, ...
                            gradOptW(idx,:), dlparams.Learnables(idx,:));
                    end
                    
                    [dlparams.Learnables,avgG.OptW,avgGS.OptW] = ...
                        adamupdate(dlparams.Learnables, gradOptW, ...
                        avgG.OptW, avgGS.OptW, globalIter, ...
                        lr, settings.beta1, settings.beta2);
            end
        end

        % Total loss.
        lossTrain = mean(double(dlUnprocess(loss, settings)));
        avgLoss = (beta*avgLoss) + ((1 - beta)*lossTrain);
        smoothedLoss = avgLoss/(1 - (beta^globalIter));

        % Plot training progress and compute auxiliary figures.
        addpoints(lossFigLine,lr,smoothedLoss);
        
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
    
    % Print out epoch update.
    elapsedTime = toc(startEpoch);
    disp(['Epoch ' num2str(i) '. '...
        'Time taken for epoch = ' num2str(elapsedTime) 's. '...
        'Learning rate = ' num2str(lr)]);
    
    % Save figure after each epoch.
    saveas(lossFig,[settings.outFolder '/' 'progress.png']);
    saveas(lossFig,[settings.outFolder '/' 'progress.fig']);
end

end

