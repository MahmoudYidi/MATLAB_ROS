function err = calcSo3Error(sixdimPred,quatTrue)
% calcSo3Error
%
% Computes the true SO(3) error.

nObs = size(sixdimPred,2);
err = zeros(1,nObs);

for i = 1:nObs
    
    % Convert predicted 6D to quaternion.
    rotmatPred = sixdimToRotmat(sixdimPred(:,i));
    quatPred = so3_to_su2(rotmatPred);
    
    % Quaternion product and error.
    dq = su2_product(quatTrue(:,i), su2_conj(quatPred));
    err_i = 2*acosd(dq(4));
    
    if ~isreal(err_i)   % Account for the floating point errors around 1 or -1
        err_i = real(err_i);
    end
    
    if err_i > 180
        err_i = 360 - err_i;
    end
    
    err(i) = err_i;
end

end

