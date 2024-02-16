function learnRate = lrDecayer(epoch,globalIter,settings,varargin)

switch lower(settings.decay)
    case 'linear'
        if epoch > settings.decayEpoch
            learnRate = settings.lr - (epoch - settings.decayEpoch)*...
                settings.lr/(settings.numEpochs - settings.decayEpoch + 1);
        else
            learnRate = settings.lr;
        end
    case 'exponential'
        exponent = globalIter/settings.decaySteps;
        if settings.staircase
            exponent = fix(exponent);
        end
        
        if epoch > settings.decayEpoch
            learnRate = settings.lr*settings.decayRate^exponent;
        else
            learnRate = settings.lr;
        end
    case 'triangular'
        if isempty(varargin)
            error('Training iterations in epoch is required for cyclical learn rate decay');
        else
            trainIterInEpoch = varargin{1};
        end
        
        stepSize = trainIterInEpoch*settings.decaySteps;
        currHalfCycle = fix(globalIter/stepSize) + 1;
        cycleIter = globalIter - (currHalfCycle - 1)*stepSize;
        decayLrBound = settings.decayLrBound;
        delta = (decayLrBound - settings.lr)/stepSize;
        
        % Ascending half-cycle
        if rem(currHalfCycle,2) ~= 0
            learnRate = settings.lr + delta*cycleIter;
            % Descending half-cycle
        else
            learnRate = decayLrBound - delta*cycleIter;
        end    
    case 'triangular2'
        if isempty(varargin)
            error('Training iterations in epoch is required for cyclical learn rate decay');
        else
            trainIterInEpoch = varargin{1};
        end
        
        stepSize = trainIterInEpoch*settings.decaySteps;
        currHalfCycle = fix(globalIter/stepSize) + 1;
        currCycle = fix((currHalfCycle-1)/2) + 1;
        cycleIter = globalIter - (currHalfCycle - 1)*stepSize;
%         decayLrBound = settings.decayLrBound/(2^(currCycle-1));
%         delta = (decayLrBound - settings.lr)/stepSize;
        decayLrBound = (settings.decayLrBound - settings.lr)/(2^(currCycle-1));
        delta = decayLrBound/stepSize;
                
        % Ascending half-cycle
        if rem(currHalfCycle,2) ~= 0
            tentativeLearnRate = settings.lr + delta*(cycleIter - 1);           
        % Descending half-cycle
        else
            tentativeLearnRate = settings.lr + decayLrBound - delta*(cycleIter - 1);    
        end
        
        learnRate = bound(tentativeLearnRate, settings.lr, settings.decayLrBound);
        
    otherwise
        learnRate = settings.lr;

end

function y = bound(x,bl,bu)
% return bounded value clipped between bl and bu
y=min(max(x,bl),bu);