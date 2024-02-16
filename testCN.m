function testCN(X)
    persistent modelStruct;
    if isempty(modelStruct)
     modelStruct = load('model45.mat');
    end
    
    % Normalize the images to [-1 1].
    X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
    
    % Convert data to dlarray specify the dimension
    % labels 'SSC' (spatial, spatial, channel).
    %dlX = dlProcess(X,'SSC',settings);
    model = modelStruct.dlnetCNN;
    % Run network.
    A = 1;
    A
    dlX = dlarray(X, 'SSC');
    dlY = squeezenetModel(dlX, model, 'test')

    % Process results.
    dlY_t = dlY(1:3);
    dlY_sixd = dlY(4:9);

    t_hat  = extractdata(dlY_t);
    sixd_hat = extractdata(dlY_sixd);

    r_hat = sixdimToRotmat(sixd_hat);
    %q_hat = so3_to_su2(r_hat);

    % Display result.
    % [t1 t2 t3 q1 q2 q3 q4]
    fprintf("%0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f\n", ...
        t_hat(1),t_hat(2),t_hat(3),...
        r_hat(1),r_hat(2),sixd_hat(3),sixd_hat(4));
   % q_hat(1),q_hat(2),q_hat(3),q_hat(4));
    
end
