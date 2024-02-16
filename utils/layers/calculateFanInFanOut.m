function [fan_in, fan_out] = calculateFanInFanOut(tensor)
% expected input: 
% [s(in) s(in) c(in) f(out)] or [f(out) s(in) s(in) c(in)]

tensor_size = num2str(length(tensor));

switch(tensor_size)
    case '4'
        num_input_fmaps = tensor(3);
        num_output_fmaps = tensor(4);
        receptive_field_size = tensor(2)*tensor(1);
    case '2'
        num_input_fmaps = tensor(2);
        num_output_fmaps = tensor(1);
        receptive_field_size = 1;
    otherwise
        error('Invalid tensor dimensions')
end

fan_in = num_input_fmaps*receptive_field_size;
fan_out = num_output_fmaps*receptive_field_size;
    
end