classdef JpegCompression
    properties
        p {mustBeNumeric}
        qualityRange
    end
    
    methods
        function obj = JpegCompression(varargin)
            % Default vals.
            obj.p            = 0.5;
            obj.qualityRange  = [2 8];
            
            if nargin > 0
                obj.p = varargin{1};
            end
            if nargin > 1
                obj.qualityRange = varargin{2};
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
                    [imgAugmented, params] = jpegCompression(obj,img,params);
                    [outIsRecord,outIsApply,outParams] = fPostApplyYes(params);
                else
                    imgAugmented = img;
                    [outIsRecord,outIsApply,outParams] = fPostApplyNo(varargin);
                end
                
                varargout{1} = {outIsRecord, outIsApply, outParams};
                
                % Otherwise, apply randomly generated parameters.
            else
                if rand > (1 - obj.p)
                    imgAugmented = jpegCompression(obj,img);
                else
                    imgAugmented = img;
                end
            end
        end
        
        function [imgAugmented, varargout] = jpegCompression(obj,img,varargin)
            % Input parsing
            nStaticArgs = 2;
            [isRandom,isRecord,params] = fFunPrep(nStaticArgs,varargin,nargin);
            
            imgAugmented = img;
            nChannels = size(img,3);
                        
            if isRandom
                qualityHyp = obj.qualityRange(1):obj.qualityRange(2);
                qualityHyp = qualityHyp(randperm(length(qualityHyp)));
                qualityVal = qualityHyp(1);
                
                if isRecord
                    params = qualityVal;
                end
            else
                qualityVal = params;
            end
            
            switch(num2str(nChannels))
                case '1'
                    imgAugmented = compress(imgAugmented,qualityVal);
                case '3'
                    imgAugmented = compressYUV(imgAugmented,qualityVal);
                case '4'
                    imgAugmented(:,:,1:3) = compressYUV(imgAugmented(:,:,1:3),qualityVal);
                    imgAugmented(:,:,4) = compress(imgAugmented(:,:,4),qualityVal);
                otherwise
                    error('No. channels not supported');
            end
            
            % Post processing.
            outParams = fFunPost(isRecord,params);
            
            varargout{1} = outParams;
        end
        
    end
end

%% Supporting functions
%  Source:
%  http://mason.gmu.edu/~mherman4/project5.html#10

function imPadded = padIm8(im)

[rowOrig, colOrig, ~] = size(im);

% Pad dimensions to make im a multiple of 8
row = 8*round(rowOrig/8);
col = 8*round(colOrig/8);

r1 = row -rowOrig;
c1 = col - colOrig;

imPadded = padarray(im, [r1, c1],'symmetric','post');

end

function imCompressed = compress(im, p)

[rowOrig, colOrig, ~] = size(im);
imPadded = padIm8(im);
[row, col, ~] = size(imPadded);

Q = p*8./hilb(8);

imPadded = double(imPadded);

for i = 1:8:row
    for j = 1:8:col
        Xb = imPadded(i:i+7,j:j+7);
        C = fun_dct(Xb);
        Xd=double(Xb);
        Xc=Xd-128;
        Y=C*Xc*C';
        Yq=round(Y./Q);
        Ydq = Yq.*Q;
        Xd = C'*Ydq*C;
        Xe=Xd+128;
        imCompressed(i:i+7,j:j+7)=Xe;
    end
end

imCompressed = uint8(rescale(imCompressed,0,255));
imCompressed = imCompressed(1:rowOrig, 1:colOrig);

end

function imCompressed = compressYUV(im, p)

[rowOrig, colOrig, ~] = size(im);
imPadded = padIm8(im);
[row, col, ~] = size(imPadded);

Q=p*[16 11 10 16 24 40 51 61; ...
    12 12 14 19 26 58 60 55; ...
    14 13 16 24 40 57 69 56; ...
    14 17 22 29 51 87 80 62; ...
    18 22 37 56 68 109 103 77; ...
    24 35 55 64 81 104 113 92; ...
    49 64 78 87 103 121 120 101; ...
    72 92 95 98 112 100 103 99];

imPadded = double(imPadded);

r = imPadded(:,:,1);
g = imPadded(:,:,2);
b = imPadded(:,:,3);

Y = 0.299*r+0.587*g+0.144*b;
U = b - Y;
V = r - Y;

for i = 1:8:row
    for j = 1:8:col
        XY = Y(i:i+7,j:j+7);
        XU = U(i:i+7,j:j+7);
        XV = V(i:i+7,j:j+7);
        
        CY = fun_dct(XY);
        CU = fun_dct(XU);
        CV = fun_dct(XV);
        XdY = double(XY);
        XdU = double(XU);
        XdV = double(XV);
        XcY = XdY-128;
        XcU = XdU-128;
        XcV = XdV-128;
        YY = CY*XcY*CY';
        YU = CU*XcU*CU';
        YV = CV*XcV*CV';
        YqY = round(YY./Q);
        YqU = round(YU./Q);
        YqV = round(YV./Q);
        YdqY = YqY.*Q;
        YdqU = YqU.*Q;
        YdqV = YqV.*Q;
        XdY = CY'*YdqY*CY;
        XdU = CU'*YdqU*CU;
        XdV = CV'*YdqV*CV;
        XeY = XdY+128;
        XeU = XdU+128;
        XeV = XdV+128;
        
        %red
        R = XeY + XeV;
        %blue
        B = XeU + XeY;
        %green
        G = (XeY - 0.299*R - 0.114*B)/0.587;
        
        imCompressed(i:i+7,j:j+7,1) = R;
        imCompressed(i:i+7,j:j+7,2) = G;
        imCompressed(i:i+7,j:j+7,3) = B;
    end
end

imCompressed = uint8(rescale(imCompressed,0,255));
imCompressed = imCompressed(1:rowOrig, 1:colOrig,:);

end

function C = fun_dct(X)

[m,n]=size(X);n=m;
for i=2:n
    for j=1:n
        C(i,j)=sqrt(2/n)*cos((i-1)*(j-1/2)*pi/n);
    end
end
C(1,1:n)=ones(1,n)/sqrt(n);

end
