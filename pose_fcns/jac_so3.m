function jac_so3_ = jac_so3(phi)

theta = norm(phi);

jac_so3_ = eye(3) + (1 - cos(theta))/theta^2*get_cross_mat(phi) + ...
    (theta - sin(theta))/theta^3*get_cross_mat(phi)^2;


end

