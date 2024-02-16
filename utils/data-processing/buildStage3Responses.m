function responses = buildStage3Responses(predictors,position,quaternion,kMat,modelPts, imSize, varargin)
% Assumptions:
%   All images are the same size
%   All images use the same K
%   modelPts is [N x 3]

nStaticArgs = 6;
nOptArgs    = 2;

optDebug    = false;
optProb     = 0.01;

if (nargin > nStaticArgs)
    optDebug = varargin{1};
end
if (nargin > nStaticArgs + 1)
    optProb = varargin{2};
end
if (nargin > nStaticArgs + nOptArgs)
    error('Number of optional arguments must be %d (provided %d)', nOptArgs, numel(varargin));
end

% Check if image should be resized.
doResize = prod(imSize > 0);

% Get number of frames.
nFrames = numel(predictors);
responses = cell(nFrames,2);

% Number of model points
nModelPts = size(modelPts,1);
% Model points in homogeneous coordinates
modelPts = [modelPts ones(nModelPts,1)];
% Flip to [4 x N];
modelPts = modelPts';

for fidx = 1:nFrames
    % Get image size.
    if (fidx == 1)
        img = imread(predictors{fidx});
        if (doResize)
           img = imresize(img, imSize); 
        end
        imSize = size(img, [1 2]);
        xSize = imSize(2);
        ySize = imSize(1);
    end
    
    % Get pose.
    p_fidx = position(fidx,:);
    q_fidx = quaternion(fidx,:);

    % Build pose matrix.
    tMat = [su2_to_so3(q_fidx') p_fidx';
            zeros(1,3)          1];
        
    % Project model points.
    if (nModelPts ~= 0)
        pt2d = kMat*tMat(1:3,:)*modelPts;
        pt2d = pt2d./pt2d(3,:);   % Normalise
        pt2d = pt2d(1:2,:);
    else
        pt2d = [];
    end
    
    % Cull points that fall outside of FOV limits
    for i = 1:nModelPts
        pt2d_x = pt2d(1,i);
        pt2d_y = pt2d(2,i);
        
        if ( (pt2d_x < 0) || (pt2d_x >= xSize) || (pt2d_y < 0) || (pt2d_y >= ySize) )
            pt2d(:,i) = nan(2,1);
        end
    end
    
    % Build response.
    responses{fidx,1} = [p_fidx q_fidx];
    responses{fidx,2} = pt2d;
    
    % Show projected points.
    if (optDebug)
        if (rand() < optProb)
            showProjectedPoints(fidx, predictors, pt2d, doResize, imSize);
        end
    end
end

end

function showProjectedPoints(fidx, predictors, pt2d, doResize, imSize)

img = imread(predictors{fidx});
if (doResize)
    img = imresize(img, imSize);
end
fig = imshow(img); hold on;

scatter(pt2d(1,:), pt2d(2,:), 'xg');
waitforbuttonpress;
close all;

end

