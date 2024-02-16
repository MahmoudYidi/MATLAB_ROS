function [ vec_ ] = vee( mat_ )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

x = (mat_(3,2) - mat_(2,3))/2;
y = (mat_(1,3) - mat_(3,1))/2;
z = (mat_(2,1) - mat_(1,2))/2;

vec_ = [x;y;z];

end

