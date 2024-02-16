function [dictionary,num_data_az,num_data_el, num_groups] = create_viewsphere_dictionary(delta)

%% Dimension checking
num_data_az = 360/delta; % number of groupings per azimuth
num_data_el = 180/delta; % number of groupings per elevation

if (mod(num_data_az, 1) ~= 0)
    error('Invalid azimuth sampling size, grid is not uniform');
end

if (mod(num_data_el, 1) ~= 0)
    error('Invalid elevation step size, grid is not uniform');
end

%% Dictionary building
num_groups = num_data_az*num_data_el;
dictionary = cell(num_groups + 1, 5);

dictionary{1,1} = 'GROUP';  dictionary{1,2} = 'AZ_MIN'; 
dictionary{1,3} = 'AZ_MAX'; dictionary{1,4} = 'EL_MIN';
dictionary{1,5} = 'EL_MAX';
for el = 1:num_data_el
    for az = 1 :num_data_az
        group = az + (el - 1)*num_data_az;
        dictionary{group + 1,1} = group;
        
        dictionary{group + 1,2} = (az - 1)*delta;
        dictionary{group + 1,3} = (az)*delta - 1;
        
        dictionary{group + 1,4} = (el - 1)*delta;
        if el ~= num_data_el
            dictionary{group + 1,5} = el*delta - 1;
        else
            dictionary{group + 1,5} = el*delta;
        end
        
    end
end

end

