classdef GaussianNoise
   properties
      p {mustBeNumeric}
      varRange
   end
   
   methods
       function obj = GaussianNoise(varargin)
           % Default vals.
           obj.p            = 0.5;
           obj.varRange  = [5e-3 1e-2];
           
           if nargin > 0
               obj.p = varargin{1};
           end
           if nargin > 1
               obj.varRange = varargin{2};
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
                   [imgAugmented, params] = gaussianNoise(obj,img,params);
                   [outIsRecord,outIsApply,outParams] = fPostApplyYes(params);
               else
                   imgAugmented = img;
                   [outIsRecord,outIsApply,outParams] = fPostApplyNo(varargin);
               end
               
               varargout{1} = {outIsRecord, outIsApply, outParams};
               
               % Otherwise, apply randomly generated parameters.
           else
               if rand > (1 - obj.p)
                   imgAugmented = gaussianNoise(obj,img);
               else
                   imgAugmented = img;
               end
           end
       end
       
       function [imgAugmented, varargout] = gaussianNoise(obj,img,varargin)
           % Input parsing
           nStaticArgs = 2;
           [isRandom,isRecord,params] = fFunPrep(nStaticArgs,varargin,nargin);
           
           imgAugmented = im2double(img);
           nChannels = size(img,3);
           nRows = size(img,1);
           nCols = size(img,2);
           
           if isRandom
               varVal = obj.varRange(1) + (obj.varRange(2) - obj.varRange(1))*rand;
               
               if isRecord
                   params = varVal;
               end
           else
               varVal = params;
           end
           
           for i = 1:nChannels
               imgAugmented(:,:,i) = imgAugmented(:,:,i) + sqrt(varVal)*randn([nRows nCols]);
           end
           
           imgAugmented = im2uint8(imgAugmented);
           
           % Post processing.
           outParams = fFunPost(isRecord,params);
           
           varargout{1} = outParams;
       end
       
   end
end