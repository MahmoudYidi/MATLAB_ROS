classdef RandomGamma
   properties
      p {mustBeNumeric}
      gammaRange
   end
   
   methods
       function obj = RandomGamma(varargin)
           % Default vals.
           obj.p            = 0.5;
           obj.gammaRange   = [80 120];
           
           if nargin > 0
               obj.p = varargin{1};
           end
           if nargin > 1
               obj.gammaRange = varargin{2};
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
                   [imgAugmented, params] = gammaJitter(obj,img,params);
                   [outIsRecord,outIsApply,outParams] = fPostApplyYes(params);
               else
                   imgAugmented = img;
                   [outIsRecord,outIsApply,outParams] = fPostApplyNo(varargin);
               end
               
               varargout{1} = {outIsRecord, outIsApply, outParams};
               
               % Otherwise, apply randomly generated parameters.
           else
               if rand > (1 - obj.p)
                   imgAugmented = gammaJitter(obj,img);
               else
                   imgAugmented = img;
               end
           end
       end
      
       function [imgAugmented, varargout] = gammaJitter(obj,img,varargin)
           % Input parsing
           nStaticArgs = 2;
           [isRandom,isRecord,params] = fFunPrep(nStaticArgs,varargin,nargin);
           
           imgAugmented = im2double(img);
           
           if isRandom
               gammaVal = obj.gammaRange(1) + (obj.gammaRange(2) - obj.gammaRange(1)).*rand;
               gammaVal = gammaVal/100;
               
               if isRecord
                   params = gammaVal;
               end
           else
               gammaVal = params;
           end
           
           imgAugmented = imgAugmented.^gammaVal;
           imgAugmented = im2uint8(imgAugmented);
           
           % Post processing.
           outParams = fFunPost(isRecord,params);
           
           varargout{1} = outParams;
       end
       
   end
end