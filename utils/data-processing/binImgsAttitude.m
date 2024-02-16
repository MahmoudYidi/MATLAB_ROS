function file_map = binImgsAttitude(filenames,q_t2c,viewsphere)

% We don't need to apply the Blender convention to categorise per se, but we keep it just in case we want to
% convert back images rendered in Blender.
% Quaternion from Blender world frame to target frame.
q_t2bworld = [0;0;0;1];
% Quaternion from Blender camera frame to OpenCV camera frame.
q_cv2bcam = [1;0;0;0];

% File structure
file_map = struct;
no_files = size(filenames,1);

% Viewsphere structure
num_data_az = viewsphere.num_data_az;
num_data_el = viewsphere.num_data_el;
delta = viewsphere.delta;

% Az/el binning
bin_edges_az = 0:delta:num_data_az*delta;
bin_edges_el = 0:delta:num_data_el*delta;

for i = 1:no_files
    % Get filename "i"
    file_map(i).filename = filenames(i,:);
    
    % Convert the quaternion mapping the target to the camera frame to viewsphere coordinates.
	%
   	% c = cv = OpenCV camera frame
    % bcam = Blender camera frame
    % bworld = Blender world frame
    % t = target frame
    %
    % q_t2bcam = q_c2bcam * q_obj2c
    % q_bworld2bcam = q_t2bcam*q_bworld2t
    %
    % We need q_bcam2bworld = q_bworld2bcam^-1.

    q_t2c_i = q_t2c(i,:);
    q_bworld2bcam = su2_product(q_cv2bcam,su2_product(q_t2c_i,su2_conj(q_t2bworld)));
    
    s_bcam2bworld = su2_to_spherical(su2_conj(q_bworld2bcam));
    az = s_bcam2bworld(1);
    el = s_bcam2bworld(2);
    
    file_map(i).az = az;
    file_map(i).el = el;
    
    az_bin = discretize(az, bin_edges_az);
    el_bin = discretize(el, bin_edges_el);
    
    group = az_bin + (el_bin - 1)*num_data_az;
    file_map(i).group = group;
end

end

