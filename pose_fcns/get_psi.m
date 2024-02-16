function psi = get_psi(q)
%GET_PSI for quaternion product. q*p = [Psi(q) q]*p

psi = [ q(4)*eye(3) - get_cross_mat(q(1:3));
        -q(1:3)'];
    
end

