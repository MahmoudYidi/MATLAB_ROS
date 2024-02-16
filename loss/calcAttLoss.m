function loss = calcAttLoss(sixdimPred,quatTrue,settings)
% calcAttLoss
%
% sixdimPred is (6 x miniBatchSize).
% quatTrue is (4 x miniBatchSize).

nObs = size(sixdimPred,2);

dims = quatTrue.dims;
loss = dlProcess(zeros(1,nObs),dims,settings);

for i = 1:nObs
    
    % Convert predicted 6D to rotation matrix.
    rotmatPred = sixdimToRotmat(sixdimPred(:,i));
    
    % Convert true quaternion to rotation matrix.
    rotmatTrue = su2_to_so3(quatTrue(:,i));
    
    % L2 norm.
    loss(i) = sqrt(dlNormFrobSq(rotmatPred - rotmatTrue));
    
end

end
