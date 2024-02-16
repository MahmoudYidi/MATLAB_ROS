function gradients = aggregateGradients(dlgradients,factor)
gradients = extractdata(dlgradients);
gradients = gplus(factor*gradients,class(gradients));
end