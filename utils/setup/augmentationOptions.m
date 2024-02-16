function settings = augmentationOptions(settings,varargin)

p = inputParser;

addRequired(p,'settings',@isstruct);

addParameter(p,'ImageRotate',false);
addParameter(p,'ImageRotateParams',{});

addParameter(p,'CamRotate',false);
addParameter(p,'CamRotateParams',{});

addParameter(p,'RandomResize',false);
addParameter(p,'RandomResizeParams',{});

addParameter(p,'RandomClahe',false);
addParameter(p,'RandomClaheParams',{});

addParameter(p,'RandomBrightnessContrast',false);
addParameter(p,'RandomBrightnessContrastParams',{});

addParameter(p,'RandomGamma',false);
addParameter(p,'RandomGammaParams',{});

addParameter(p,'ChannelShift',false);
addParameter(p,'ChannelShiftParams',{});

addParameter(p,'MedianBlur',false);
addParameter(p,'MedianBlurParams',{});

addParameter(p,'GaussianBlur',false);
addParameter(p,'GaussianBlurParams',{});

addParameter(p,'JpegCompression',false);
addParameter(p,'JpegCompressionParams',{});

addParameter(p,'GaussianNoise',false);
addParameter(p,'GaussianNoiseParams',{});

addParameter(p,'PatchDropout',false);
addParameter(p,'PatchDropoutParams',{});

parse(p,settings,varargin{:})

imAugList = fieldnames(p.Results);
imAugList = imAugList(1:end-1);     % Remove 'settings'

imAugs = {};
idx = 0;
for i = 1:numel(imAugList)

    imAug_i = imAugList{i};
    
    if (contains(imAug_i,'Params'))
        continue;
    end
    
    includeAugmentation = getImAugStatus(p,imAug_i);
    
    if (~includeAugmentation)
        continue;
    end
    
    augmentationParams = getImAugStatus(p,[imAug_i 'Params']);
    idx = idx + 1;
        
    switch(imAug_i)
        case 'ImageRotate'
            imAugs{idx} = ImageRotate(augmentationParams{:});
        case 'CamRotate'
            imAugs{idx} = CamRotate(augmentationParams{:});
        case 'RandomResize'
            imAugs{idx} = RandomResize(augmentationParams{:});
        case 'RandomClahe'
            imAugs{idx} = RandomClahe(augmentationParams{:});
        case 'RandomBrightnessContrast'
            imAugs{idx} = RandomBrightnessContrast(augmentationParams{:});
        case 'RandomGamma'
            imAugs{idx} = RandomGamma(augmentationParams{:});
        case 'ChannelShift'
            imAugs{idx} = ChannelShift(augmentationParams{:});
        case 'MedianBlur'
            imAugs{idx} = MedianBlur(augmentationParams{:});
        case 'GaussianBlur'
            imAugs{idx} = GaussianBlur(augmentationParams{:});
        case 'JpegCompression'
            imAugs{idx} = JpegCompression(augmentationParams{:});
        case 'GaussianNoise'
            imAugs{idx} = GaussianNoise(augmentationParams{:});
        case 'PatchDropout'
            imAugs{idx} = PatchDropout(augmentationParams{:});
        otherwise
            continue;
    end
end

settings.imageAugmentation = imAugs;

end

function val = getImAugStatus(parser,imAugName) %#ok<INUSL>
    val = eval(sprintf('%s.Results.%s;','parser',imAugName));
end

    