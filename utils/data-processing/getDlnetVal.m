function val = getDlnetVal(tab,layer,param)
% getDlnetVal
%
% Looks up a corresponding value in a custom dlnet learnables table.
%
% The table must be organised in Table Layer, Parameter, and Value
% variables, in this order.
%
% Value must be a cell array of dlarrays.

idx = (tab.Layer == layer & tab.Parameter == param);
idx = find(idx==1);
val = tab.Value{idx};
end

