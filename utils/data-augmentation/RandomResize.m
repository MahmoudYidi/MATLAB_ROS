classdef RandomResize
    properties
        p {mustBeNumeric}
        resizeRange
    end
    
    methods
        function obj = RandomResize(varargin)
            % Default vals.
            obj.p            = 0.5;
            obj.resizeRange   = [0.8 1.2];
            
            if nargin > 0
                obj.p = varargin{1};
            end
            if nargin > 1
                obj.resizeRange = varargin{2};
            end
        end
        
        function imgAugmented = apply(obj, img)
            if rand > (1 - obj.p)
                imgAugmented = randomResize(obj,img);
            else
                imgAugmented = img;
            end
        end
        
        function imgAugmented = randomResize(obj,img)
            resizeVal = obj.resizeRange(1) + (obj.resizeRange(2) - obj.resizeRange(1))*rand;
            
            imgAugmented = imresize(img,resizeVal);
        end
    end
end