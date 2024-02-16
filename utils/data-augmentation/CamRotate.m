classdef CamRotate
    properties
        p {mustBeNumeric}
        magnitude
    end
    
    methods
        function obj = CamRotate(varargin)
            % Default vals.
            obj.p           = 0.5;
            obj.magnitude   = 20;
            
            if nargin > 0
                obj.p = varargin{1};
            end
            if nargin > 1
                obj.magnitude = varargin{2};
            end
        end
        
        function [imgAugmented, varargout] = apply(obj, img, isConsistent, varargin)
            % Initialise output variables.
            varargout{1} = {};  % Params
            varargout{2} = {};  % Modified pose
            
            % Get ground parameters.
            K = varargin{2}{1};     % Intrinsic
            f1 = varargin{2}{2};    % Flag: modify pose
            f2 = varargin{2}{3};    % Flag: have model points
            y = varargin{2}{4};     % Pose label(s)
                        
            if isConsistent
                % Apply consistently with same parameters.
                [isApply,params] = fPrepConsistent(obj,varargin); 	% Input check
                
                if isApply
                    [imgAugmented, params, gtOut] = camRotate(obj,img,K,y,f1,f2,params);
                    [outIsRecord,outIsApply,outParams] = fPostApplyYes(params);
                else
                    imgAugmented = img;
                    [outIsRecord,outIsApply,outParams] = fPostApplyNo(varargin);
                end
                
                varargout{1} = {outIsRecord, outIsApply, outParams};
                varargout{2} = gtOut;
                
                % Otherwise, apply randomly generated parameters.
            else
                if rand > (1 - obj.p)
                    [imgAugmented, ~, gtOut] = camRotate(obj,img,K,y,f1,f2);
                    varargout{2} = gtOut;
                else
                    imgAugmented = img;
                end
            end
        end
        
        function [imgAugmented, varargout] = camRotate(obj,img,K,y,fModPose,fModPts,varargin)
            % Input parsing
            nStaticArgs = 6;
            
            if fModPose
                pose = y{1};
                t = pose(1:3);
                q = pose(4:7);
                if fModPts
                    pts = y{2};
                end
            end
            
            [isRandom,isRecord,params] = fFunPrep(nStaticArgs,varargin,nargin);
            
            % Apply transformation.
            if isRandom                
                eulChange = (rand(1,3) - 0.5).*obj.magnitude;
                
                if isRecord
                    params = eulChange;
                end
            else
                eulChange = params;
            end
            
            R_change = euler_to_so3(eulChange(1),eulChange(2), eulChange(3));
            M = K*R_change/K;
            
            tform = projective2d(M');           

            imReference = imref2d(size(img));
            imgAugmented = imwarp(img, tform, 'OutputView', imReference);
            
            % Update pose.
            t_new = t*R_change';
            q_change = so3_to_su2(R_change);
            q_new = su2_product(q_change, q);
            pose_new = [t_new q_new'];
            
            % Modify projected model points according to the homography.
            if (fModPts)
                nPts = size(pts,2);
                ptsH = [pts; ones(1,nPts)];
                ptsM = M*ptsH;
                ptsM = ptsM(1:2,:)./ptsM(3,:);
                
                % Check if any point falls out of bounds.
                ptsContained = contains(imReference,ptsM(1,:),ptsM(2,:));
                idxPtsOut = find(ptsContained == false);
                if ~isempty(idxPtsOut)
                    ptsM(:, idxPtsOut) = nan;
                end
            end
            
            % Post processing.
            outParams = fFunPost(isRecord,params);
            
            % Output argument assignment.
            varargout{1} = outParams;
            
            varargout{2} = cell(1, length(y));
            varargout{2}{1} = pose_new;
            if fModPts
                varargout{2}{2} = {pose_new,ptsM};
            end
        end
        
    end
end
