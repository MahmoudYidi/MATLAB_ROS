function [imAugmented, imAugParamsOut, gtParamsOut] = imageAugmenter(im,imAugStruct,isConsistent, imAugParams, gtParams)

imAugParamsOut = {};
gtParamsOut = {};

bModifiedPose = false;

imAugmented = im;

for i = 1:numel(imAugStruct)
    if isConsistent
        imAugParams_i = imAugParams{i};
    else
        imAugParams_i = {};
    end
    
    [imAugmented, status, gtOut] = imAugStruct{i}.apply(imAugmented, isConsistent, imAugParams_i, gtParams);
    
    % Parameter update.
    if ~isempty(status)
        isRecord = status{1};
        if isRecord
            imAugParams{i} = {status{2}, status{3}};
        end
    end
    
    % Ground truth update.
    if ~isempty(gtOut)
        bModifiedPose = true;
        gtParams{end} = gtOut;
    end
end

if isConsistent
   imAugParamsOut = imAugParams; 
end

if bModifiedPose
    gtParamsOut = gtParams;
end

end

