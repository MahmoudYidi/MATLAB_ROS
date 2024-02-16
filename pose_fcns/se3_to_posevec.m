function rt = se3_to_posevec(T)
    R =  T(1:3,1:3);
    t  = T(1:3, 4);
    
    rv = log_so3(R);
      
    rt = [t;
          rv];    
end

