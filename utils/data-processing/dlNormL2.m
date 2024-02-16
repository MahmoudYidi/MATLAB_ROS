function norm = dlNormL2(dlX)

norm = 0;
for i = 1:length(dlX)
   norm = norm + dlX(i)^2; 
end

norm = sqrt(norm);

end

