function pt2d = reproject3dPoint(TMat,KMat,pt3d)
% reproject3dPoint
%
% Based on an extrinsic (TMat) and intrinsic (KMat) camera matrix, repro-
% ject the 3D point (pt3d)
%
% TMat          SE(3) pose matrix                   [4x4]
% KMat          Camera calibration matrix           [3x3]
% pt3d          3D point                            [3x1]

pt3d = [pt3d; 1];               % homogenous coordinates
pt2d = KMat*TMat(1:3,:)*pt3d;
pt2d = pt2d./pt2d(3);           % normalise
pt2d = pt2d(1:2);

end

