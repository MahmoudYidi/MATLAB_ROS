function ad_ = ad_se3(screw_)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

rho = screw_(1:3);
phi = screw_(4:6);

phi_cross_mat = get_cross_mat(phi);
rho_cross_mat = get_cross_mat(rho);

ad_ = [phi_cross_mat,   rho_cross_mat;
       zeros(3,3),      phi_cross_mat];
   
end

