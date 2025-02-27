function out = su2_to_so3(su2_)
%SU2_TO_SO3
%    OUT = SU2_TO_SO3(E1,E2,E3,Q1)

%    This function was generated by the Symbolic Math Toolbox version 7.2.
%    19-Feb-2019 15:16:20

e1 = su2_(1);
e2 = su2_(2);
e3 = su2_(3);
q1 = su2_(4);

t2 = e1.*e2.*2.0;
t3 = e3.*q1.*2.0;
t4 = e1.^2;
t5 = e2.^2;
t6 = e3.^2;
t7 = q1.^2; 
t8 = e1.*e3.*2.0;
t9 = e2.*e3.*2.0;
t10 = e1.*q1.*2.0;
out = reshape([t4-t5-t6+t7,t2-t3,t8+e2.*q1.*2.0,t2+t3,-t4+t5-t6+t7,t9-t10,t8-e2.*q1.*2.0,t9+t10,-t4-t5+t6+t7],[3,3]);
