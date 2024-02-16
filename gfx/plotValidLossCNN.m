function lineLossValid = plotValidLossCNN(lossValid, lineLossValid, epoch, numIter)
% plotValidLoss
%
% Plots the validation loss for one epoch.

lossTotal = get2TableVal(lossValid,"TotalLoss");
addpoints(lineLossValid{1}.loss.valid,epoch*numIter,lossTotal);
addpoints(lineLossValid{2}.loss.valid,epoch*numIter,lossTotal);

lossPos = get2TableVal(lossValid,"PositionLoss");
addpoints(lineLossValid{3}.loss.valid,epoch*numIter,lossPos);

lossAtt = get2TableVal(lossValid,"AttitudeLoss");
addpoints(lineLossValid{4}.loss.valid,epoch*numIter,lossAtt);
end