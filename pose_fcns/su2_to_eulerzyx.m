function eulervec = su2_to_eulerzyx(q)

e1 = q(1);
e2 = q(2);
e3 = q(3);
q0 = q(4);

ez = atan2(-2*e1*e2 + 2*q0*e3, e1^2 + q0^2 - e3^2 - e2^2);
ey = asin(2*e1*e3 + 2*q0*e2);
ex = atan2(-2*e2*e3 + 2*q0*e2, e3^2 - e2^2 - e1^2 + q0^2);

eulervec = [ez;ey;ex];

end

