function viewsphere = create_viewsphere(delta)

viewsphere.delta = delta;   % Viewsphere patch sampling size for training
[viewsphere.dictionary,viewsphere.num_data_az,viewsphere.num_data_el,...
    viewsphere.num_groups] = ...
        create_viewsphere_dictionary(delta);    % Mapping between group no.
                                                % and viewsphere coordi-
                                                % nates

end

