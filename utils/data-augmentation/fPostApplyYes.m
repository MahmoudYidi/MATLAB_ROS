function [outIsRecord,outIsApply,outParams] = fPostApplyYes(params)

if ~isempty(params)
    outIsRecord = true;
    outIsApply = true;
    outParams = params;
else
    outIsRecord = false;
    outIsApply = {};
    outParams = {};
end

end

