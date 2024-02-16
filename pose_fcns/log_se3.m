function screw_ = log_se3( se3_ )
%LOG_SE3 Summary of this function goes here
%   Detailed explanation goes here

r_ = se3_(1:3,1:3);
t_ = se3_(1:3, 4);

phi = log_so3(r_);
theta = norm(phi);
V = eye(3) + (1 - cos(theta))/theta^2*get_cross_mat(phi) + ...
    (theta - sin(theta))/theta^3*get_cross_mat(phi)^2;

rho = inv(V)*t_;

screw_ = [rho; phi];

end

