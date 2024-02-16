function screw_ = log_pose_su2( pose_su2_ )
%LOG_SE3 Summary of this function goes here
%   Detailed explanation goes here

q_ = pose_su2_(4:7);
t_ = pose_su2_(1:3);

phi = log_su2(q_);
theta = norm(phi);
V = eye(3) + (1 - cos(theta))/theta^2*get_cross_mat(phi) + ...
    (theta - sin(theta))/theta^3*get_cross_mat(phi)^2;

rho = inv(V)*t_;

screw_ = [rho; phi];

end

