classdef RandomBrightnessContrast
    properties
        p {mustBeNumeric}
        brightnessRange
        contrastRange
    end
    
    methods
        function obj = RandomBrightnessContrast(varargin)
            % Default vals.
            obj.p               = 0.5;
            obj.contrastRange   = [0.8 1.2];
            obj.brightnessRange = [-0.2 0.2];
            
            if nargin > 0
                obj.p = varargin{1};
            end
            if nargin > 1
                obj.brightnessRange = varargin{2};
            end
            if nargin > 2
                obj.contrastRange = varargin{3};
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
                    [imgAugmented, params] = brightnessContrastJitter(obj,img,params);
                    [outIsRecord,outIsApply,outParams] = fPostApplyYes(params);
                else
                    imgAugmented = img;
                    [outIsRecord,outIsApply,outParams] = fPostApplyNo(varargin);
                end
                
                varargout{1} = {outIsRecord, outIsApply, outParams};
                
                % Otherwise, apply randomly generated parameters.
            else
                if rand > (1 - obj.p)
                    imgAugmented = brightnessContrastJitter(obj,img);
                else
                    imgAugmented = img;
                end
            end
        end
        
        function [imgAugmented, varargout] = brightnessContrastJitter(obj,img,varargin)
            % Input parsing
            nStaticArgs = 2;
            [isRandom,isRecord,params] = fFunPrep(nStaticArgs,varargin,nargin);
            
            imgAugmented = im2double(img);
            
            if isRandom
                brightnessVal = obj.brightnessRange(1) + (obj.brightnessRange(2) - obj.brightnessRange(1)).*rand;
                contrastVal = obj.contrastRange(1) + (obj.contrastRange(2) - obj.contrastRange(1)).*rand;
                
                if isRecord
                    params = {brightnessVal,contrastVal};
                end
            else
                brightnessVal = params{1};
                contrastVal = params{2};
            end
            
            imgAugmented = imgAugmented.*contrastVal + brightnessVal;
            imgAugmented = im2uint8(imgAugmented);
            
            % Post processing.
            outParams = fFunPost(isRecord,params);
            
            varargout{1} = outParams;
        end
        
    end
end