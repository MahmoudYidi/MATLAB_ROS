function [q] = spherical2quat(az,el)

az_rad = deg2rad(az);
el_rad = deg2rad(el);

ex = -pi/2 - az_rad;
ey = -pi/2 + el_rad;
ez = pi/2;

q = eulerzyx_to_su2([ez;ey;ex]);

end

