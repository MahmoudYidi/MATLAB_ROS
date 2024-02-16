function [alpha, n] = su2_to_axisangle(q)
% e1 = q(1);
% e2 = q(2);
% e3 = q(3);
q0 = q(4);

alpha = 2*acos(q0);

if abs(alpha) < eps('single')
    n = nan(3,1);
else
    n = q(1:3)/norm(q(1:3));
end

end

