function R = euler_to_so3(pitch, yaw, roll)

cos_pitch = cos(pitch*pi/180);
sin_pitch = sin(pitch*pi/180);
cos_yaw = cos(yaw*pi/180);
sin_yaw = sin(yaw*pi/180);
cos_roll = cos(roll*pi/180);
sin_roll = sin(roll*pi/180);

R = [cos_yaw*cos_roll, sin_pitch*sin_yaw*cos_roll - cos_pitch*sin_roll, cos_pitch*sin_yaw*cos_roll + sin_pitch*sin_roll;
     cos_yaw*sin_roll, sin_pitch*sin_yaw*sin_roll + cos_pitch*cos_roll, cos_pitch*sin_yaw*sin_roll - sin_pitch*cos_roll;
     -sin_yaw, sin_pitch*cos_yaw, cos_pitch*cos_yaw];

end

