function gradients = thresholdAbsoluteValue(gradients,gradientThreshold)
% https://uk.mathworks.com/help/deeplearning/ug/specify-training-options-in-custom-training-loop.html

gradients(gradients > gradientThreshold) = gradientThreshold;
gradients(gradients < -gradientThreshold) = -gradientThreshold;

end