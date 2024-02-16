function distMat = buildAttitudeLossDictionary(dictionary, mult)

nClasses = size(dictionary,1) - 1;
sphericalCoords = zeros(nClasses,2);    % Longitude (lambda), latitude (phi)
distMat = zeros(nClasses, nClasses);    % Distance matrix

% Compute patch centres.
for i = 1:nClasses
    [az, el] = findPatchCentre(i, dictionary);
    sphericalCoords(i, 1) = az2lon(az);
    sphericalCoords(i, 2) = el2lat(el);
end

% Build distance matrix.
switch true
    
    case mult > 0
        for i = 1:nClasses
            lambda_1 = sphericalCoords(i,1);    % Correct longitude
            phi_1 = sphericalCoords(i,2);       % Correct latitude
            
            cphi1 = cosd(phi_1);
            sphi1 = sind(phi_1);
            
            for j = 1:nClasses
                lambda_2 = sphericalCoords(j,1);
                phi_2 = sphericalCoords(j,2);
                
                dlambda = lambda_2 - lambda_1;
                cphi2 = cosd(phi_2);
                sphi2 = sind(phi_2);
                
                num = (cphi2*sind(dlambda))^2 + (cphi1*sphi2 - sphi1*cphi2*cosd(dlambda))^2;
                num = sqrt(num);
                den = sphi1*sphi2 + cphi1*cphi2*cosd(dlambda);
                
                dist = atan2d(num,den);
                distMat(j,i) = exp(-dist/mult);
            end
        end
        
    case mult == -1
        distMat = eye(nClasses);
        
    otherwise
        error('Wrong ''mult'' value');

end

end

function [az,el] = findPatchCentre(class, dictionary)
% Assume groups are sorted by 'ascend'

row = class + 1;
az_min = dictionary{row,2};
az_max = dictionary{row,3};
el_min = dictionary{row,4};
el_max = dictionary{row,5};

az = round((az_max - az_min)/2) + az_min;
el = round((el_max - el_min)/2) + el_min;
end

function lon = az2lon(az)

if (az <= 180) && (az >= 0)
    lon = az;
else
    lon = az - 360;
end

end

function lat = el2lat(el)
    lat = (el - 90)*(-1);
end