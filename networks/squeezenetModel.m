function dlY = squeezenetModel(dlX, dlnet, varargin)
%% Input parsing
if numel(varargin) == 0
    status = 'train';
else
    status = varargin{1};
end

%% Conv1
%  conv2d -> relu -> maxpool
dlY = dlX;

convId = "conv1";

W = getDlnetVal(dlnet.Learnables,convId,"Weights");
b = getDlnetVal(dlnet.Learnables,convId,"Bias");
p = getDlnetVal(dlnet.Operators,convId,"Padding");
s = getDlnetVal(dlnet.Operators,convId,"Stride");
dlY = dlconv(dlY,W,b,'Stride',s,'Padding',p);

lreluId = "lrelu1";
s = getDlnetVal(dlnet.Operators,lreluId,"Scale");
dlY = leakyrelu(dlY,s);

poolId = "pool1";

k = getDlnetVal(dlnet.Operators,poolId,"Kernel");
p = getDlnetVal(dlnet.Operators,poolId,"Padding");
s = getDlnetVal(dlnet.Operators,poolId,"Stride");

dlY = maxpool(dlY, k, 'Stride', s, 'Padding', p);

%% Fire2
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu)
%                               ∟-------- concat ----------┘
fireId = ["fire2/squeeze1x1";"fire2/lrelu_squeeze1x1";
    "fire2/expand1x1";"fire2/lrelu_expand1x1";
    "fire2/expand3x3";"fire2/lrelu_expand3x3"];
dlYf2 = fireBlock(dlY, dlnet, fireId);

%% Fire3
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu)
%                               ∟-------- concat ----------┘
fireId = ["fire3/squeeze1x1";"fire3/lrelu_squeeze1x1";
    "fire3/expand1x1";"fire3/lrelu_expand1x1";
    "fire3/expand3x3";"fire3/lrelu_expand3x3"];
dlYf3 = fireBlock(dlYf2, dlnet, fireId);
%  Skip connection
dlY = dlYf2 + dlYf3;

%% Fire4
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu) -> maxpool
%                               ∟-------- concat ----------┘
fireId = ["fire4/squeeze1x1";"fire4/lrelu_squeeze1x1";
    "fire4/expand1x1";"fire4/lrelu_expand1x1";
    "fire4/expand3x3";"fire4/lrelu_expand3x3"];
dlY = fireBlock(dlY, dlnet, fireId);

poolId = "pool4";

k = getDlnetVal(dlnet.Operators,poolId,"Kernel");
p = getDlnetVal(dlnet.Operators,poolId,"Padding");
s = getDlnetVal(dlnet.Operators,poolId,"Stride");

dlYf4 = maxpool(dlY, k, 'Stride', s, 'Padding', p);

%% Fire5
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu)
%                               ∟-------- concat ----------┘
fireId = ["fire5/squeeze1x1";"fire5/lrelu_squeeze1x1";
    "fire5/expand1x1";"fire5/lrelu_expand1x1";
    "fire5/expand3x3";"fire5/lrelu_expand3x3"];
dlYf5 = fireBlock(dlYf4, dlnet, fireId);
%  Skip connection
dlY = dlYf4 + dlYf5;

%% Fire6
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu)
%                               ∟-------- concat ----------┘
fireId = ["fire6/squeeze1x1";"fire6/lrelu_squeeze1x1";
    "fire6/expand1x1";"fire6/lrelu_expand1x1";
    "fire6/expand3x3";"fire6/lrelu_expand3x3"];
dlYf6 = fireBlock(dlY, dlnet, fireId);

%% Fire7
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu)
%                               ∟-------- concat ----------┘
fireId = ["fire7/squeeze1x1";"fire7/lrelu_squeeze1x1";
    "fire7/expand1x1";"fire7/lrelu_expand1x1";
    "fire7/expand3x3";"fire7/lrelu_expand3x3"];
dlYf7 = fireBlock(dlYf6, dlnet, fireId);
%  Skip connection
dlY = dlYf6 + dlYf7;

%% Fire8
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu) -> maxpool
%                               ∟-------- concat ----------┘
fireId = ["fire8/squeeze1x1";"fire8/lrelu_squeeze1x1";
    "fire8/expand1x1";"fire8/lrelu_expand1x1";
    "fire8/expand3x3";"fire8/lrelu_expand3x3"];
dlY = fireBlock(dlY, dlnet, fireId);

poolId = "pool8";

k = getDlnetVal(dlnet.Operators,poolId,"Kernel");
p = getDlnetVal(dlnet.Operators,poolId,"Padding");
s = getDlnetVal(dlnet.Operators,poolId,"Stride");

dlYf8 = maxpool(dlY, k, 'Stride', s, 'Padding', p);

%% Fire9
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu) -> dropout
%                               ∟-------- concat ----------┘
fireId = ["fire9/squeeze1x1";"fire9/lrelu_squeeze1x1";
    "fire9/expand1x1";"fire9/lrelu_expand1x1";
    "fire9/expand3x3";"fire9/lrelu_expand3x3"];
dlYf9 = fireBlock(dlYf8, dlnet, fireId);
%  Skip connection
dlY = dlYf8 + dlYf9;
%  Dropout
if strcmp(status,'train')
    p = getDlnetVal(dlnet.Operators,"dropout9","Rate");
    dlY = dropout(dlY,p);
end

%% Conv10
%  conv2d -> relu -> maxpool
convId = "conv10";

W = getDlnetVal(dlnet.Learnables,convId,"Weights");
b = getDlnetVal(dlnet.Learnables,convId,"Bias");
p = getDlnetVal(dlnet.Operators,convId,"Padding");
s = getDlnetVal(dlnet.Operators,convId,"Stride");
dlY = dlconv(dlY,W,b,'Stride',s,'Padding',p);

lreluId = "lrelu10";
s = getDlnetVal(dlnet.Operators,lreluId,"Scale");
dlY = leakyrelu(dlY,s);

dlY = globalAvgPool(dlY);

end

function dlY = fireBlock(dlX, dlnet, blockIds)

s1x1Id  = blockIds{1};
rs1x1Id = blockIds{2};
e1x1Id  = blockIds{3};
re1x1Id = blockIds{4};
e1x3Id  = blockIds{5};
re3x3Id = blockIds{6};

% Squeeze 1x1
W = getDlnetVal(dlnet.Learnables,s1x1Id,"Weights");
b = getDlnetVal(dlnet.Learnables,s1x1Id,"Bias");
p = getDlnetVal(dlnet.Operators,s1x1Id,"Padding");
s = getDlnetVal(dlnet.Operators,s1x1Id,"Stride");
dlY = dlconv(dlX,W,b,'Stride',s,'Padding',p);

s = getDlnetVal(dlnet.Operators,rs1x1Id,"Scale");
dlY = leakyrelu(dlY,s);

% Expand 1x1
W = getDlnetVal(dlnet.Learnables,e1x1Id,"Weights");
b = getDlnetVal(dlnet.Learnables,e1x1Id,"Bias");
p = getDlnetVal(dlnet.Operators,e1x1Id,"Padding");
s = getDlnetVal(dlnet.Operators,e1x1Id,"Stride");
dlY1 = dlconv(dlY,W,b,'Stride',s,'Padding',p);

% disp(re1x1Id);
s = getDlnetVal(dlnet.Operators,re1x1Id,"Scale");
dlY1 = leakyrelu(dlY1,s);

% Expand 3x3
W = getDlnetVal(dlnet.Learnables,e1x3Id,"Weights");
b = getDlnetVal(dlnet.Learnables,e1x3Id,"Bias");
p = getDlnetVal(dlnet.Operators,e1x3Id,"Padding");
s = getDlnetVal(dlnet.Operators,e1x3Id,"Stride");
dlY2 = dlconv(dlY,W,b,'Stride',s,'Padding',p);

s = getDlnetVal(dlnet.Operators,re3x3Id,"Scale");
dlY2 = leakyrelu(dlY2,s);

% Concatenate
channelDim = find(dlX.dims == 'C');
dlY = cat(channelDim, dlY1, dlY2);

end