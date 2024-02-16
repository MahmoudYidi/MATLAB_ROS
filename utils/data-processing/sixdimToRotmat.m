function rotmat = sixdimToRotmat(sixdim)
% 6 x 1 -> 3 x 3 (rotation matrices)
% @inproceedings{zhou_continuity_2019,
%   title = {On the {{Continuity}} of {{Rotation Representations}} in {{Neural Networks}}},
%   booktitle = CVPR,
%   author = {Zhou, Yi and Barnes, Connelly and Lu, Jingwan and Yang, Jimei and Li, Hao},
%   year = {2019},
%   pages = {9}
% }

sixdim = reshape(sixdim, [3 2]);
x_raw = sixdim(:,1);
y_raw = sixdim(:,2);

x = x_raw/dlNormL2(x_raw);
z = cross(x, y_raw);
z = z/dlNormL2(z);
y = cross(z,x);

rotmat = [x y z];

end