classdef RandomClahe
    properties
        p {mustBeNumeric}
        clipRange
        gridSize
    end
    
    methods
        function obj = RandomClahe(varargin)
            % Default vals.
            obj.p       	= 0.5;
            obj.clipRange 	= [1,4];
            obj.gridSize    = [8,8];
            
            if nargin > 0
                obj.p = varargin{1};
            end
            if nargin > 1
                obj.clipRange = varargin{2};
            end
            if nargin > 2
                obj.gridSize = varargin{3};
            end
        end
        
        function [imgAugmented, varargout] = apply(obj, img,isConsistent, varargin)
            % Initialise output variables.
            varargout{1} = {};  % Params
            varargout{2} = {};  % Modified pose
            
            if isConsistent
                % Apply consistently with same parameters.
                [isApply,params] = fPrepConsistent(obj, varargin); 	% Input check
                
                if isApply
                    [imgAugmented, params] = randomClahe(obj,img,params);
                    [outIsRecord,outIsApply,outParams] = fPostApplyYes(params);
                else
                    imgAugmented = img;
                    [outIsRecord,outIsApply,outParams] = fPostApplyNo(varargin);
                end
                
                varargout{1} = {outIsRecord, outIsApply, outParams};
                
                % Otherwise, apply randomly generated parameters.
            else
                if rand > (1 - obj.p)
                    imgAugmented = randomClahe(obj,img);
                else
                    imgAugmented = img;
                end
            end
        end
        
        function [imgAugmented, varargout] = randomClahe(obj,img,varargin)
            % Input parsing
            nStaticArgs = 2;
            [isRandom,isRecord,params] = fFunPrep(nStaticArgs,varargin,nargin);
            
            imgAugmented = im2double(img);
            nChannels = size(img,3);
                        
            if isRandom
                clipVal = obj.clipRange(1) + (obj.clipRange(2)-obj.clipRange(1))*rand;
                clipVal = clipVal/256;
                
                if isRecord
                    params = clipVal;
                end
            else
                clipVal = params;
            end
            
            switch(num2str(nChannels))
                case '1'
                    imgAugmented = adapthisteq(imgAugmented,'clipLimit',clipVal,'NumTiles',obj.gridSize);
                case '3'
                    imgAugmented = adapthisteqColour(imgAugmented, obj.gridSize, clipVal);
                case '4'
                    imgAugmented(:,:,1:3) = adapthisteqColour(imgAugmented(:,:,1:3), obj.gridSize, clipVal);
                    imgAugmented(:,:,4) = adapthisteq(imgAugmented(:,:,4),'clipLimit',clipVal,'NumTiles',obj.gridSize);
                otherwise
                    error('No. channels not supported');
            end
            
            imgAugmented = im2uint8(imgAugmented);
            
            % Post processing.
            outParams = fFunPost(isRecord,params);
            
            varargout{1} = outParams;
        end
        
    end
end

function imClahe  = adapthisteqColour(im, numTiles, clipLimit)

lab = rgb2lab(im);

L = lab(:,:,1)/100;

L = adapthisteq(L,'NumTiles',numTiles,'ClipLimit',clipLimit);
lab(:,:,1) = L*100;

imClahe = lab2rgb(lab);
end

