function [ vec_out ] = vec(mat_in)
    [r,c] = size(mat_in);
    %vec_out = zeros(r*c,1);
    vec_out = [];
    
    for i = 1:c
       %jj = i*r;
       %vec_out(jj - r + 1:jj) = mat_in(:,i);
       vec_out = [vec_out; mat_in(:,i)];
    end
end

