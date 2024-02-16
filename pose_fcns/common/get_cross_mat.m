function [ vec_x ] = get_cross_mat( x )

vec_x = [0 -x(3) x(2);
         x(3) 0 -x(1);
         -x(2) x(1) 0];

end

