function jac_se3_ = jac_se3(xi)

rho = xi(1:3);
phi = xi(4:6);

nphi = norm(phi);
rho_c = get_cross_mat(rho);
phi_c = get_cross_mat(phi);

B1 = eye(3) + (1 - cos(nphi))*phi_c/nphi^2 + (nphi - sin(nphi))*phi_c^2/nphi^3;
B2 = 0.5*rho_c + ((nphi - sin(nphi))/nphi^3)*(phi_c*rho_c + rho_c*phi_c + phi_c*rho_c*phi_c) +...
    ((nphi^2 + 2*cos(nphi) - 2)/(2*nphi^4))*(phi_c*phi_c*rho_c + rho_c*phi_c*phi_c - 3*phi_c*rho_c*phi_c) +...
    ((2*nphi - 3*sin(nphi) + nphi*cos(nphi))/(2*nphi^5))*(phi_c*rho_c*phi_c*phi_c + phi_c*phi_c*rho_c*phi_c);

jac_se3_ = [B1            B2;
            zeros(3,3)    B1];

end

