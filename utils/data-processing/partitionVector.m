function partitions = partitionVector(seqLen, sampler, minLen)

startIdx    = [];
endIdx      = [];
elements    = [];
group       = [];

j = 1;
k = 1;
framesLeft = seqLen;

while (framesLeft > 0)
    slen_exp_i = sampler.random;
    slen_i = 2^slen_exp_i;
    
    frames_left_next = framesLeft - slen_i;
    if frames_left_next < 0
        slen_i_eff = framesLeft;
    else
        slen_i_eff = slen_i;
    end
    framesLeft = frames_left_next;
    
    if (slen_i_eff >= minLen)
        startIdx = [startIdx; k];
        endIdx = [endIdx; k + slen_i_eff - 1];
        elements = [elements; slen_i_eff];
        group = [group; j];
    else
        endIdx(end) = endIdx(end) + slen_i_eff;
        elements(end) = elements(end) + slen_i_eff;
    end
        
    k = k + slen_i_eff;
    j = j + 1;
end

partitions = table(group, elements, startIdx, endIdx, ...
    'VariableNames', ["Group","Elements","Start","End"]);

end

