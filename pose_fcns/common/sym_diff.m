function [ out_mat ] = sym_diff(a, b)
    [rows_a, cols_a] = size(a);
    [rows_b, cols_b] = size(b);
    
    assert(cols_a == 1, 'cols_a != 1');
    assert(cols_b == 1, 'cols_b != 1');
    
    out_mat = sym('m', [rows_a, rows_b], 'real');
    
    for i = 1:rows_a
        for j = 1:rows_b
            out_mat(i,j) = diff(a(i),b(j));
        end
    end

end

