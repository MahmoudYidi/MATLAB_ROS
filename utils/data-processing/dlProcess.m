function dlX = dlProcess(x,label,settings)

dlX = dlarray(x, label);

% If training on a GPU, then convert data to gpuArray.
if settings.useGPU == true
    dlX = gpuArray(dlX);
end

end

