function norm = dlNormFrobSq(dlX)

norm = 0;
dlX = dlX'*dlX;
for i = 1:length(dlX)
   norm = norm + dlX(i,i); 
end

end

