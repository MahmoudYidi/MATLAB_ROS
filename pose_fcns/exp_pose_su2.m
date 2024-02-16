function [ pose_su2_ ] = exp_pose_su2( screw )

rho = screw(1:3);
phi = screw(4:6);
theta = norm(phi);

aux1 = exp_su2(phi);

V = eye(3) + (1 - cos(theta))/theta^2*get_cross_mat(phi) + ...
    (theta - sin(theta))/theta^3*get_cross_mat(phi)^2;

pose_su2_ = [V*rho; aux1];

end

