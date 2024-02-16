function gradients = thresholdL2Norm(gradients,gradientThreshold)
% https://uk.mathworks.com/help/deeplearning/ug/specify-training-options-in-custom-training-loop.html
gradientNorm = sqrt(sum(gradients(:).^2));
if gradientNorm > gradientThreshold
    gradients = gradients * (gradientThreshold / gradientNorm);
end

end