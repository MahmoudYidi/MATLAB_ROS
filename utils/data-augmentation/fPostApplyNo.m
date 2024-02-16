function [outIsRecord,outIsApply,outParams] = fPostApplyNo(argsin)

if isempty(argsin{1})
    outIsRecord = true;
    outIsApply = false;
    outParams = {};
else
    outIsRecord = false;
    outIsApply = {};
    outParams = {};
end

end

