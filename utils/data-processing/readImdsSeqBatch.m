function [pred,resp,varargout] = readImdsSeqBatch(imdsTable,imdsGroup,idx,settings,...
    doImageAugmentation, doImAugConsistency, varargin)

% Argument parsing
nStaticArgs = 6;
%nOptArgs = 2;

imAugParams = {};   % Image augmenter params, in case of consistency
gtParams = {};      % Ground truth parameters required by certain transforms, in this order:
                    %   - K 	intrinsic matrix
                    %   - f1 	modify pose flag
                    %   - f2    have model points flag

if (nargin > nStaticArgs)
    imAugParams = varargin{1};
end
if (nargin > nStaticArgs + 1)
    gtParams = varargin{2};
end

% Struct building
startIdx = imdsGroup.Start(idx);
endIdx = imdsGroup.End(idx);
readSz = endIdx - startIdx + 1;

% Read minibatch
pred = cell(readSz,1);
resp = cell(readSz,1);
%datas = cell(readSz,1);

for i = 0:readSz-1
    % Read data pair
    pred_i = readImdsSeq(imdsTable, startIdx+i);
    
    % Data pre-processing
    predAug_i = preprocessImds(pred_i,settings);
    
    % Read responses
    resp_i = imdsTable.Responses(startIdx+i,:);
    gtParams_i = gtParams;
    gtParams_i{end+1} = resp_i;
    
    % Data augmentation
    if doImageAugmentation
        
        [predAug_i, imAugParams, gtOut] = imageAugmenter(predAug_i,settings.imageAugmentation, doImAugConsistency, ...
            imAugParams, gtParams_i);

%         figure;
%         montage({predAug_i(:,:,1:3), predAug_i(:,:,4)});
%         waitforbuttonpress;

        if ~isempty(gtOut)
            resp_i = gtOut{end};
        end
    end
    
    % Read dataset number
    %datas_i = imdsTable.Dataset(startIdx+i,1);
    
    % Store
    pred{i+1} = predAug_i;
    resp{i+1} = resp_i;
    %datas{i+1} = datas_i;
end

if doImAugConsistency
   varargout{1} = imAugParams; 
end

end

