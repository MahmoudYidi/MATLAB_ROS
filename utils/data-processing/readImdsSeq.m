function data = readImdsSeq(imdsTable,idx)
%
filename_i = imdsTable.Predictors(idx,:);

nImgs = size(filename_i, 2);

if nImgs == 1
    data = imread(filename_i{1});
else
    data = [];
    
    for i = 1:nImgs
        data_i = imread(filename_i{i});
        data = cat(3, data, data_i);
    end
end

end

