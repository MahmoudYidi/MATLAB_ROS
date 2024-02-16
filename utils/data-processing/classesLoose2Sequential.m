function [dictionaryNew, classesNew] = classesLoose2Sequential(dictionary, classes)

dictionaryNew   = dictionary;
classesNew      = 1:numel(classes); % New, contiguous classes

i = 1;
while i < numel(classesNew) + 1
    class_i = dictionaryNew{i+1,1};
    if (class_i == classes(i))   % curClasses is sorted
        dictionaryNew{i+1,1} = classesNew(i);
        i = i + 1;
        
        if i == numel(classesNew) + 1
            dictionaryNew(i+1:end,:) = [];
        end
        
        continue;
    else
        dictionaryNew(i+1,:) = [];
    end
end

end