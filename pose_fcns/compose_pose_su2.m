function [ out ] = compose_pose_su2(lhs, rhs)

aux1 = su2_product(lhs(4:7), rhs(4:7));
aux2 = compose_pose_point_su2(lhs, rhs(1:3));

out = [aux2; aux1];
end

