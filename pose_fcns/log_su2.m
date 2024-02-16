function [ phi ] = log_su2( su2_ )
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
e_ = su2_(1:3);
norm_e = norm(e_);
q_ = su2_(4);

phi = -2*acos(q_)/norm_e*e_;

end

