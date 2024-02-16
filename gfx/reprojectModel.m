function imgOut = reprojectModel(imgIn,TMat,KMat,modelPts)
% reprojectModel
%
% Based on an extrinsic (TMat) and intrinsic (KMat) camera matrix, repro-
% ject the 3D model points (modelPts) onto an input image (imgIn).
%
% TMat          SE(3) pose matrix                   [4x4]
% KMat          Camera calibration matrix           [3x3]
% modelPts      n 3D model points of target object  [nx12]

noPts = size(modelPts,1);
imgOut = imgIn;

for i = 1:noPts
   pt_i = modelPts(i,:);
   pt_i = reshape(pt_i, [4 3]);
   
   for j = 1:4  % each modelPt must be a closed 4-sided polygon
       pt_ij = pt_i(j,:);
       pt_ik = pt_i(mod(j+1-1,4)+1,:);
       
       lineStart = reproject3dPoint(TMat,KMat,pt_ij');
       lineEnd = reproject3dPoint(TMat,KMat,pt_ik');
       
       imgOut = insertShape(imgOut,'Line',[lineStart' lineEnd'],...
           'LineWidth',2,'Color','green');
   end
   
end

end

