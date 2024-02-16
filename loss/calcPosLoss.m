function loss = calcPosLoss(yPred,yTrue)
% calcPosLoss
%
% {yPred, yTrue} are (3 x miniBatchSize).

% Compute loss.
diff = yPred - yTrue;
loss = sum(diff.^2,1);
loss = sqrt(loss);

end
