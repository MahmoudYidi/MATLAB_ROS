function T = posevec_to_se3(rt)
    
    t = rt(1:3);
    rv = rt(4:6);

    R = exp_so3(rv);
    T = [R              t;
         zeros(1,3)     1];
end