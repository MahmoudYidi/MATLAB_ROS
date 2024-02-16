function [isApply,params] = fPrepConsistent(obj,argsin)

% Input check
if ~isempty(argsin{1})              % Params passed; use them
    isApply = argsin{1}{1};
    params = argsin{1}{2};
else                                % No params passed; initialise
    isApply = rand > (1 - obj.p);
    params = {};
end

end

