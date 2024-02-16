function [file_map,bin_edges] = binImgsDistance(filenames,p_t2c,p_min,p_max,p_delta)

% File structure
file_map = struct;
no_files = size(filenames,1);

% Binning
bin_edges = p_min:p_delta:p_max;

for i = 1:no_files
    % Get filename "i"
    file_map(i).filename = filenames(i,:);

    p_i = p_t2c(i,:);
    dist = norm(p_i);
    
    if (dist < p_min)
        dist = p_min;
    elseif (dist > p_max)
        dist = p_max;
    end
    
    p_bin = discretize(dist, bin_edges);
%     disp(p_bin);
    file_map(i).group = p_bin;
end

end

