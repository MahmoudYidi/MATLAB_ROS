function q = eulerzyx_to_su2(eulervec)

ez = eulervec(1);
ey = eulervec(2);
ex = eulervec(3);

qz = axisangle_to_su2(ez, [0;0;1]);
qy = axisangle_to_su2(ey, [0;1;0]);
qx = axisangle_to_su2(ex, [1;0;0]);

q = su2_product(su2_product(qz,qy),qx);

end

