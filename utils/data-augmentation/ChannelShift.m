classdef ChannelShift
    properties
        p {mustBeNumeric}
        shiftLimit
        isCustom
    end
    
    methods
        function obj = ChannelShift(varargin)
            % Default vals.
            obj.p            = 0.5;
            obj.shiftLimit   = 20;
            obj.isCustom     = false;
            
            if nargin > 0
                obj.p = varargin{1};
            end
            if nargin > 1
                obj.shiftLimit = varargin{2};
                obj.isCustom = true;
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
                    [imgAugmented, params] = shiftChannel(obj,img,params);
                    [outIsRecord,outIsApply,outParams] = fPostApplyYes(params);
                else
                    imgAugmented = img;
                    [outIsRecord,outIsApply,outParams] = fPostApplyNo(varargin);
                end
                
                varargout{1} = {outIsRecord, outIsApply, outParams};
                
                % Otherwise, apply randomly generated parameters.
            else
                if rand > (1 - obj.p)
                    imgAugmented = shiftChannel(obj,img);
                else
                    imgAugmented = img;
                end
            end
        end
        
        function [imgAugmented, varargout] = shiftChannel(obj,img,varargin)
            % Input parsing
            nStaticArgs = 2;
            [isRandom,isRecord,params] = fFunPrep(nStaticArgs,varargin,nargin);
            
            imgAugmented = im2double(img);
            nChannels = size(img,3);
            
            if ~obj.isCustom
                shiftLimit = obj.shiftLimit*ones(1,nChannels); %#ok<*PROPLC>
            else
                shiftLimit = obj.shiftLimit;
            end
            
            for i = 1:nChannels
                if isRandom
                    shiftVal = -shiftLimit(i) + 2*shiftLimit(i).*rand;
                    shiftVal = shiftVal/256;
                    
                    if isRecord
                        params{i} = shiftVal;
                    end
                else
                    shiftVal = params{i};
                end
                
                imgAugmented(:,:,i) = imgAugmented(:,:,i) + shiftVal;
            end
            
            imgAugmented = im2uint8(imgAugmented);
            
            % Post processing.
            outParams = fFunPost(isRecord,params);
            
            varargout{1} = outParams;
        end
        
    end
end