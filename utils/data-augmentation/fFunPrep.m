function [isRandom,isRecord,params] = fFunPrep(nStaticArgs,argsin,numargsin)

isRandom = true;
isRecord = false;
params = {};

if numargsin > nStaticArgs
    params = argsin{1};
    
    if isempty(params)
        isRandom = true;
        isRecord = true;
        params = cell(1,3);
    else
        isRandom = false;
        isRecord = false;
    end
end

end

