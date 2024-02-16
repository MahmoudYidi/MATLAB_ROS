function q = axisangle_to_su2(alpha, n)

q0 = cos(0.5*alpha);
e = n*sin(0.5*alpha);

q = [e;q0];

end

