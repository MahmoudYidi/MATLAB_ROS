function [ out ] = compose_rot_point_su2(lhs, rhs)

aux = su2_product(lhs, su2_product([rhs;0], su2_conj(lhs)));
out = aux(1:3);

end

