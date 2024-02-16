function [ pose_su2_inv_ ] = pose_su2_inv( pose_su2_ )
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here

aux1 = su2_conj(pose_su2_(4:7));
aux2 = compose_rot_point_su2(aux1, -pose_su2_(1:3));

pose_su2_inv_ = [aux2; aux1];

end

