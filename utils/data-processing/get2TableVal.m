function val = get2TableVal(tab,param)
% get2TableVal
%
% Looks up a corresponding value in a custom two-column table.
%
% The table must be organised in Table Parameter and Value
% variables, in this order.

idx = tab.Parameter == param;
idx = find(idx==1);
val = tab.Value(idx);
end

