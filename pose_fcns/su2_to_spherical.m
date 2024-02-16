function [spherical] = su2_to_spherical(quat)
% float e1 = this->vec.at<float>(0,0);
% float e2 = this->vec.at<float>(1,0);
% float e3 = this->vec.at<float>(2,0);
% float q = this->scalar;
e1 = quat(1);
e2 = quat(2);
e3 = quat(3);
q = quat(4);

% Convert first to Euler angles (ZYX-extrinsic, like Blender).
% R = Rx(r_x)*Ry(r_y)*Rz(r_z)

% float det = e1*e3 - e2*q;
det = e1*e3 - e2*q;

% if (fabs(fabs(det) - 0.5) < Manifolds::tol) {
%   det > 0 ? y_r = M_PI/2.f : y_r = -M_PI/2.f;
%   x_r = std::atan2(2.f*(e2*e3 - e1*q), 1 - 2.f*(pow(e2,2) + pow(q,2)));
% }
% else {
%   y_r = std::asin(2.f*det);
%   x_r = std::atan2(2.f*(e2*e3 + e1*q), 1 - 2.f*(pow(e3,2) + pow(q,2)));
% }
if (abs(abs(det) - 0.5) < eps('single'))
    y_r = pi/2;
    if (det <= 0)
        y_r = -y_r;
    end
    
    x_r = atan2(2*(e2*e3 - e1*q), 1-2*(e2^2 + q^2));
else
   y_r = asin(2*det);
   x_r = atan2(2*(e2*e3 + e1*q), 1-2*(e3^2 + q^2));
end

% x_r += M_PI;
x_r = x_r + pi;

% u = [atan2(-2*e1*e2 + 2*q*e3, e1^2 + q^2 - e3^3 - e2^2);
%      asin(2*e1*e3 + 2*q*e2);
%      atan2(-2*e2*e3 + 2*q*e1, e3^2-e2^2-e1^2 + q^2)];
% u = [atan2(2*e2*e3 + 2*q*e1, e3^2 - e2^2 - e1^3 + q^2);
%      -asin(2*e1*e3 - 2*q*e2);
%      atan2(2*e1*e2 + 2*q*e3, e1^2+q^2-e3^2 - e2^2)];
 
% u = quat2eul([q e1 e2 e3], 'XYZ');
% 
% y_r = u(2);
% x_r = u(1);


% Convert to viewsphere coordinates, according to Blender script.
% az = -pi/2 + x_r
% el = y_r - pi/2

% float az = -M_PI/2.f - x_r;
% float el = y_r + M_PI/2.f;
az = -pi/2 - x_r;
el = y_r + pi/2;

az = rad2deg(az);
el = rad2deg(el);

if (az >= 360 || az < 0)
    az = wrap_angle(az);
end

el = wrap_angle(el);

if (el > 180)
    el = 360 - el;
    az = az + 180;
end

spherical = [az;el];

end

function wrapped_angle = wrap_angle(angle)
wrapped_angle = angle - 360*floor(angle/360);
end

