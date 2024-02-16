function [err_t, err_r] = calcSe3Error(yPred,yTrue)
% calcSe3Error
%
% Computes the true SE(3) error.

nObs = size(yPred,2);
err_t = zeros(1,nObs);
err_r = zeros(1,nObs);

for i = 1:nObs
    
    % Convert from se(3) to SU(2) x R3
    su2Pred = yPred(:,i);
    su2True = yTrue(:,i);    
    
    % Position
    tPred = su2Pred(1:3);
    tTrue = su2True(1:3);
    err_t(i) = norm(tPred - tTrue);
    
    % Orientation
    qPred = su2Pred(4:7);
    qTrue = su2True(4:7);
    qPredNorm = norm(qPred);    % Normalise quaternion
    qPred = qPred/qPredNorm;
    dq = su2_product(qTrue, su2_conj(qPred));
    
    err_r_i = 2*acosd(dq(4));
    
    if err_r_i > 180
        err_r_i = 360 - err_r_i;
    end
    
    err_r(i) = err_r_i;
    
end

end

