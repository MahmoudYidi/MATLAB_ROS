function lossValid = calcValidLossArray(dataValid, gt, dlnet, settings)
% calcValidLossArray
%
% Computes the validation loss for one epoch when the data is in the form of an array.

% Define the desired types of losses
lossLabels = ["Mse";"TranslationError";"RotationError"];

dlX = dlProcess(dataValid,'CT',settings);

YTrue = gt;
dlYTrue = dlProcess(YTrue,'CT',settings);

dlYPred = lstmNetModel(dlX, dlnet, 'test');
YPred = dlUnprocess(dlYPred, settings);

% losses/errors
lossMseArray = calcSe3MseLoss(dlYPred,dlYTrue,settings);
[err_tArray, err_rArray] = calcSe3Error(YPred,YTrue);

% Average the results
lossMseValid = mean(double(dlUnprocess(lossMseArray,settings)));
err_tValid = mean(err_tArray);
err_rValid = mean(err_rArray);

% Compile into a single table
lossValid = table(lossLabels, [lossMseValid;err_tValid;err_rValid], ...
    'VariableNames', ["Parameter","Value"]);

end

