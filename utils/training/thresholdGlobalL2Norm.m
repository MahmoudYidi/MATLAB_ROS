function gradients = thresholdGlobalL2Norm(gradients,gradientThreshold)
% https://uk.mathworks.com/help/deeplearning/ug/specify-training-options-in-custom-training-loop.html

globalL2Norm = 0;
for i = 1:numel(gradients)
    globalL2Norm = globalL2Norm + sum(gradients{i}(:).^2);
end
globalL2Norm = sqrt(globalL2Norm);

if globalL2Norm > gradientThreshold
    normScale = gradientThreshold / globalL2Norm;
    for i = 1:numel(gradients)
        gradients{i} = gradients{i} * normScale;
    end
end

end