function settings = modeSetup(settings)
%  Flag for GPU use
if (settings.executionEnvironment == "auto" && canUseGPU) || settings.executionEnvironment == "gpu"
    if (settings.useParallel == true)
        numberOfGPUs = gpuDeviceCount;
        
        if (numberOfGPUs > 1)
            pool = parpool(numberOfGPUs);
            settings.useGPU = true;
        else
            settings.useGPU = false;
            pool = parpool;
            %             pool = parpool(2); % PARALLEL GPU TEST
        end
    else
        settings.useGPU = true;
    end
else
    settings.useGPU = false;
    
    if (settings.useParallel == true)
        pool = parpool;
    end
end

if (settings.useParallel == true)
    N = pool.NumWorkers;
    
    if settings.useGPU == true
        settings.miniBatchSize = settings.miniBatchSize*N;
    end
    
    workerMiniBatchSize = floor(settings.miniBatchSize./repmat(N,1,N));
    remainder = settings.miniBatchSize - sum(workerMiniBatchSize);
    workerMiniBatchSize = workerMiniBatchSize + [ones(1,remainder) zeros(1,N-remainder)];
    
    settings.numWorkers = N;
    settings.workerMiniBatchSize = workerMiniBatchSize;
end
end

