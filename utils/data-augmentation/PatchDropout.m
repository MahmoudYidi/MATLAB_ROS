classdef PatchDropout
    properties
        p {mustBeNumeric}
        pPatch
        patchRange
    end
    
    methods
        function obj = PatchDropout(varargin)
            % Default vals.
            obj.p            = 0.5;
            obj.pPatch       = 0.10;
            obj.patchRange   = [3 5];
            
            if nargin > 0
                obj.p = varargin{1};
            end
            if nargin > 1
                obj.pPatch = varargin{2};
            end
            if nargin > 2
                obj.patchRange = varargin{3};
            end
        end
        
        function [imgAugmented, varargout] = apply(obj, img, isConsistent, varargin)
            % Initialise output variables.
            varargout{1} = {};  % Params
            varargout{2} = {};  % Modified pose
            
            if isConsistent
                % Apply consistently with same parameters.
                [isApply,params] = fPrepConsistent(obj, varargin); 	% Input check
                
                if isApply
                    [imgAugmented, params] = patchDropout(obj,img,params);
                    [outIsRecord,outIsApply,outParams] = fPostApplyYes(params);
                else
                    imgAugmented = img;
                    [outIsRecord,outIsApply,outParams] = fPostApplyNo(varargin);
                end
                
                varargout{1} = {outIsRecord, outIsApply, outParams};
                
                % Otherwise, apply randomly generated parameters.
            else
                if rand > (1 - obj.p)
                    imgAugmented = patchDropout(obj,img);
                else
                    imgAugmented = img;
                end
            end
        end
        
        function [imgAugmented, varargout] = patchDropout(obj,img,varargin)
            % Input parsing
            nStaticArgs = 2;
            [isRandom,isRecord,params] = fFunPrep(nStaticArgs,varargin,nargin);
            
            imgAugmented = im2double(img);
            
            nRows = size(img,1);
            nCols = size(img,2);
            minDim = min(nRows,nCols);
                      
            if isRandom
                patchSize = obj.patchRange(1) + (obj.patchRange(2) - obj.patchRange(1))*rand;
                patchSize = patchSize/100;
                
                if isRecord
                    params = patchSize;
                end
            else
                patchSize = params;
            end
            
            patchPx = round(patchSize*minDim);
            
            dropoutMat = true(nRows,nCols);
            nXDivs = ceil(nCols/patchPx);
            nYDivs = ceil(nRows/patchPx);
            
            for i = 1:nYDivs
                startLimY = (i-1)*patchPx + 1;
                
                if (i == nYDivs)
                    endLimY = nRows;
                else
                    endLimY = startLimY + patchPx - 1;
                end
                
                for j = 1:nXDivs
                    startLimX = (j-1)*patchPx + 1;
                    
                    if (j == nXDivs)
                        endLimX = nCols;
                    else
                        endLimX = startLimX + patchPx - 1;
                    end
                    
                    if rand < obj.pPatch
                        dropoutMat(startLimY:endLimY, startLimX:endLimX) = false;
                    end
                end
            end
            
            imgAugmented = imgAugmented.*dropoutMat;
            imgAugmented = im2uint8(imgAugmented);
            
            % Post processing.
            outParams = fFunPost(isRecord,params);
            
            varargout{1} = outParams;
        end
        
    end
end