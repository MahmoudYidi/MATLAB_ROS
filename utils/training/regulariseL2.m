function grad = regulariseL2(fcName,dlnet,grad,settings)

condStr = sprintf("settings.l2Reg.%s.do == true", fcName);
condEval = eval(condStr);

if (condEval)
    lambdaStr = sprintf("settings.l2Reg.%s.lambda", fcName);
    l2RegLambda = eval(lambdaStr);
    idx = dlnet.Learnables.Parameter == "Weights";
    grad(idx,:) = dlupdate(@(g,w) g + l2RegLambda*w, ...
        grad(idx,:), dlnet.Learnables(idx,:));
end

end

