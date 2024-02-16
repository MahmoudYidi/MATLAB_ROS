function imgAugmented = preprocessImds(img,settings)
%
imgAugmented    = img;
imSize          = settings.imSize;
doResize        = prod(imSize > 0);

if doResize
    imgAugmented = imresize(imgAugmented,settings.imSize);
end

end

