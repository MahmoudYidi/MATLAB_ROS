function x = dlUnprocess(dlX,settings)

x = extractdata(dlX);

if settings.useGPU == true
    x = gather(x);
end

end

