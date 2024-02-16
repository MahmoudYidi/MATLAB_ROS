function network = squeezenetParams(hyperparameters)

%% Initialise
[network,tmpTreeLearnables] = layerInitialiser();

%% Conv1
%  conv2d -> relu -> maxpool
stride          = 2;
padding         = 0;
kernel          = 7;
nChannelsIn     = hyperparameters.nChannelsIn;
nChannelsOut    = 96;
leakyReluScale  = 0.1;

tmpTreeLearnables = buildConv2d("conv1", hyperparameters, tmpTreeLearnables, ...
                                kernel, stride, padding, ...
                                [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
                                'fan_in', 'leaky_relu',leakyReluScale);
                            
tmpTreeLearnables = buildLeakyRelu("lrelu1", tmpTreeLearnables, leakyReluScale);

kernel = 3;
stride = 2;
padding = 0;
                            
tmpTreeLearnables = buildMaxPool("pool1", tmpTreeLearnables, kernel, stride, padding);

%% Fire2
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu)
%                               ∟-------- concat ----------┘
blockId = "fire2/squeeze1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 96;
nChannelsOut    = 16;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
                                kernel, stride, padding, ...
                                [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
                                'fan_in', 'leaky_relu', leakyReluScale);
                            
tmpTreeLearnables = buildLeakyRelu("fire2/lrelu_squeeze1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire2/expand1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 16;
nChannelsOut    = 64;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
                                kernel, stride, padding, ...
                                [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
                                'fan_in', 'leaky_relu', leakyReluScale);
                            
tmpTreeLearnables = buildLeakyRelu("fire2/lrelu_expand1x1", tmpTreeLearnables, leakyReluScale);
                            
blockId = "fire2/expand3x3";
                            
stride          = 1;
padding         = 1;
kernel          = 3;
nChannelsIn     = 16;
nChannelsOut    = 64;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
                    kernel, stride, padding, ...
                    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
                    'fan_in', 'leaky_relu', leakyReluScale);
                
tmpTreeLearnables = buildLeakyRelu("fire2/lrelu_expand3x3", tmpTreeLearnables, leakyReluScale);
                
%% Fire3
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu)
%                               ∟-------- concat ----------┘
blockId = "fire3/squeeze1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 128;
nChannelsOut    = 16;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
                                kernel, stride, padding, ...
                                [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
                                'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire3/lrelu_squeeze1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire3/expand1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 16;
nChannelsOut    = 64;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
                                kernel, stride, padding, ...
                                [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
                                'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire3/lrelu_expand1x1", tmpTreeLearnables, leakyReluScale);                            
                            
blockId = "fire3/expand3x3";
                            
stride          = 1;
padding         = 1;
kernel          = 3;
nChannelsIn     = 16;
nChannelsOut    = 64;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
                    kernel, stride, padding, ...
                    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
                    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire3/lrelu_expand3x3", tmpTreeLearnables, leakyReluScale);                
%% Fire4
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu) -> maxpool
%                               ∟-------- concat ----------┘
blockId = "fire4/squeeze1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 128;
nChannelsOut    = 32;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire4/lrelu_squeeze1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire4/expand1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 32;
nChannelsOut    = 128;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire4/lrelu_expand1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire4/expand3x3";

stride          = 1;
padding         = 1;
kernel          = 3;
nChannelsIn     = 32;
nChannelsOut    = 128;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire4/lrelu_expand3x3", tmpTreeLearnables, leakyReluScale);

kernel = 3;
stride = 2;
padding = 0;

tmpTreeLearnables = buildMaxPool("pool4", tmpTreeLearnables, kernel, stride, padding);

%% Fire5
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu)
%                               ∟-------- concat ----------┘
blockId = "fire5/squeeze1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 256;
nChannelsOut    = 32;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire5/lrelu_squeeze1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire5/expand1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 32;
nChannelsOut    = 128;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire5/lrelu_expand1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire5/expand3x3";

stride          = 1;
padding         = 1;
kernel          = 3;
nChannelsIn     = 32;
nChannelsOut    = 128;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire5/lrelu_expand3x3", tmpTreeLearnables, leakyReluScale);

%% Fire6
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu)
%                               ∟-------- concat ----------┘
blockId = "fire6/squeeze1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 256;
nChannelsOut    = 48;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire6/lrelu_squeeze1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire6/expand1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 48;
nChannelsOut    = 192;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire6/lrelu_expand1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire6/expand3x3";

stride          = 1;
padding         = 1;
kernel          = 3;
nChannelsIn     = 48;
nChannelsOut    = 192;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire6/lrelu_expand3x3", tmpTreeLearnables, leakyReluScale);

%% Fire7
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu)
%                               ∟-------- concat ----------┘
blockId = "fire7/squeeze1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 384;
nChannelsOut    = 48;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire7/lrelu_squeeze1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire7/expand1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 48;
nChannelsOut    = 192;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire7/lrelu_expand1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire7/expand3x3";

stride          = 1;
padding         = 1;
kernel          = 3;
nChannelsIn     = 48;
nChannelsOut    = 192;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire7/lrelu_expand3x3", tmpTreeLearnables, leakyReluScale);

%% Fire8
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu)
%                               ∟-------- concat ----------┘
blockId = "fire8/squeeze1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 384;
nChannelsOut    = 64;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire8/lrelu_squeeze1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire8/expand1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 64;
nChannelsOut    = 256;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire8/lrelu_expand1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire8/expand3x3";

stride          = 1;
padding         = 1;
kernel          = 3;
nChannelsIn     = 64;
nChannelsOut    = 256;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire8/lrelu_expand3x3", tmpTreeLearnables, leakyReluScale);

kernel = 3;
stride = 2;
padding = 0;

tmpTreeLearnables = buildMaxPool("pool8", tmpTreeLearnables, kernel, stride, padding);

%% Fire9
%  squeeze1x1 -> relu -> (expand1x1 -> relu) | (expand3x3 -> relu) -> dropout
%                               ∟-------- concat ----------┘
blockId = "fire9/squeeze1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 512;
nChannelsOut    = 64;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire9/lrelu_squeeze1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire9/expand1x1";

stride          = 1;
padding         = 0;
kernel          = 1;
nChannelsIn     = 64;
nChannelsOut    = 256;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire9/lrelu_expand1x1", tmpTreeLearnables, leakyReluScale);

blockId = "fire9/expand3x3";

stride          = 1;
padding         = 1;
kernel          = 3;
nChannelsIn     = 64;
nChannelsOut    = 256;

tmpTreeLearnables = buildConv2d(blockId, hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("fire9/lrelu_expand3x3", tmpTreeLearnables, leakyReluScale);

dropoutRate = 0.5;
tmpTreeLearnables = buildDropout("dropout9", tmpTreeLearnables, dropoutRate);

%% Conv10
%  conv2d -> relu -> gap
stride          = 1;
padding         = 1;
kernel          = 1;
nChannelsIn     = 512;
nChannelsOut    = hyperparameters.nChannelsOut;

tmpTreeLearnables = buildConv2d("conv10", hyperparameters, tmpTreeLearnables, ...
    kernel, stride, padding, ...
    [kernel,kernel,nChannelsIn,nChannelsOut], [1,1,nChannelsOut],...
    'fan_in', 'leaky_relu', leakyReluScale);

tmpTreeLearnables = buildLeakyRelu("lrelu10", tmpTreeLearnables, leakyReluScale);
                        
%% Post-process
network.Learnables = table(tmpTreeLearnables.netLayer, tmpTreeLearnables.netParam, tmpTreeLearnables.netVal', ...
    'VariableNames', ["Layer","Parameter","Value"]);
network.Operators = table(tmpTreeLearnables.opLayer, tmpTreeLearnables.opParam, tmpTreeLearnables.opVal, ...
    'VariableNames', ["Layer","Parameter","Value"]);

end



