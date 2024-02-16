function [ phi_ ] = log_so3( so3_ )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
theta = acos((trace(so3_) - 1)/2);
% theta = acos((so3_(1,1)+so3_(2,2)+so3_(3,3) - 1)/2);

%log_R = theta/(2*sin(theta))*(so3_ - so3_');
if (abs(theta) <  2.5e-4)
    isinc = 0.5;
else
    isinc = theta/(2*sin(theta));
end

log_R = isinc*(so3_ - so3_');
    
phi_ = vee(log_R);

end

