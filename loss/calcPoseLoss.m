function [Lp, Lq] = calcPoseLoss(yPred,yTrue)
% calcSe3Mse
%
% Computes an L1 pose loss.
% Pose format: position + quaternion.
%
% {yPred, yTrue} are (7 x miniBatchSize).

% Split position and orientation.
pTrue = yTrue(1:3,:);   % Position
pPred = yPred(1:3,:);

qTrue = yTrue(4:7,:);   % Quaternion
qPred = yPred(4:7,:);

% Compute position L1 loss.
Lp = sum(abs(pTrue - pPred),1);

% Compute quaternion L1 loss.
% Limit the quaternions to one hemisphere.
qTrue = limitSu2(qTrue);
qPred = limitSu2(qPred);

% Normalise the predicted quaternions.
for i = 1:size(qPred,2)
    qPred(:,i) = qPred(:,i)./dlNormL2(qPred(:,i));
end

% L1 loss.
Lq = sum(abs(qTrue - qPred),1);

end

function q_limited = limitSu2(q)
% Limits the quaternions to one hemisphere.
% Default: positive scalar component.

negScalars = q(4,:) < 0;     % Scalar components
q_limited = q;
q_limited(negScalars) = q_limited(negScalars)*-1;

end

