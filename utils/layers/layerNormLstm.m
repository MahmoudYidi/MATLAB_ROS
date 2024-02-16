function [X, h0, c0] = layerNormLstm(X, h0, c0, W, R, B, a1, a2, a3, b3, varargin)
%LAYERNORMLSTM   Layer-normalised long short-term memory
%
%    Y = LSTM(X,H0,C0,WEIGHTS,RECURRENTWEIGHTS,BIAS) applies a long
%    short-term memory (lstm) calculation to the labeled dlarray X, using
%    initial hidden state H0 and initial cell state C0. WEIGHTS,
%    RECURRENTWEIGHTS, and BIAS are learnable parameters for the operation.
%
%    X must be a labeled dlarray containing a dimension labeled 'T'. If X
%    has any dimensions labeled 'S', they are flattened into the 'C'
%    dimension. If X has any dimensions labeled 'U', they must be
%    singleton. If X has a dimension labeled 'B', it is unaffected by the
%    lstm operation.
%    
%    Inputs H0, C0, WEIGHTS, RECURRENTWEIGHTS, and BIAS can be labeled
%    dlarrays, unlabeled dlarrays, or numeric arrays. Output Y is a labeled
%    dlarray. Except for any 'S' dimensions, Y has the same dimension
%    labels as X.
%
%    Inputs H0 and C0 must be vectors or matrices. If H0 or C0 are labeled
%    dlarrays, then they must contain the label 'C' and optionally the
%    label 'B'. The size of the 'C' dimension determines the number of
%    hidden units, and must be the same for H0 and C0. If H0 or C0 are
%    unlabeled dlarrays or numeric arrays, the size of the first dimension
%    determines the number of hidden units.
%
%    Input WEIGHTS must be a matrix of size 4*NumHiddenUnits-by-InputSize,
%    where NumHiddenUnits is given by the size of the 'C' dimension of H0
%    and InputSize is the size of the 'C' dimension of X. If WEIGHTS is a
%    labeled dlarray, then it must contain a 'C' dimension of size
%    4*NumHiddenUnits and a 'U' dimension of size InputSize.
%
%    Input RECURRENTWEIGHTS must be a matrix of size
%    4*NumHiddenUnits-by-NumHiddenUnits. If RECURRENTWEIGHTS is a labeled
%    dlarray, then it must contain a 'C' dimension of size 4*NumHiddenUnits
%    and a 'U' dimension of size NumHiddenUnits.
%
%    Input BIAS must be a vector of size 4*NumHiddenUnits. If BIAS is a
%    labeled dlarray, then it must be labeled with 'C'.
%
%    [Y,HIDDENSTATE,CELLSTATE] =
%    LSTM(X,H0,C0,WEIGHTS,RECURRENTWEIGHTS,BIAS) applies an lstm
%    calculation to the labeled dlarray X and also returns the hidden state
%    and cell state. HIDDENSTATE and CELLSTATE have the same type as H0 and
%    C0, respectively. If the inputs H0 or C0 are labeled dlarrays, outputs
%    HIDDENSTATE and CELLSTATE are labeled 'CB'.
%
%    [Y,HIDDENSTATE,CELLSTATE] =
%    LSTM(X,H0,C0,WEIGHTS,RECURRENTWEIGHTS,BIAS,'DataFormat',FMT) applies
%    an lstm calculation to the unlabeled dlarray or a numeric array X. FMT
%    specifies the dimension labels of the input X. FMT must be a char
%    array or string. If the input X is a numeric array, at least one of
%    H0, C0, WEIGHTS, RECURRENTWEIGHTS, or BIAS must be a dlarray. The
%    output Y is an unlabeled dlarray with the same dimension order as X.
%
%    Copyright   2019    The MathWorks, Inc.
%    Modified    2020    Duarte Rondao

% Note: it is assumed that the dlarray is 'CBT' or 'CT'.
nStaticArgs = 10;
nOptArgs = 4;

stride = -1;
if nargin > nStaticArgs
    stride = varargin{1};
end

mode = 'train';
if nargin > nStaticArgs + 1
    mode = varargin{2};
end

pZoneH = 0;
if nargin > nStaticArgs + 2
    pZoneH = varargin{3};
end

pZoneC = 0;
if nargin > nStaticArgs + 3
    pZoneC = varargin{4};
end

if nargin > nStaticArgs + nOptArgs
    error('Wrong number of optional arguments');
end

% Retrieve channel, time and observation dimensions
timeDim = find(X.dims == 'T');
channelDim = find(X.dims == 'C');
observationDim = find(X.dims == 'B');

% Ensure there is a T dimension, a C dimension, and no non-singleton U dimensions
hasNoTimeDim = isempty(timeDim);
hasNoChannelDim = isempty(channelDim);
XHasNoObservationDim = isempty(observationDim);
sizeXData = size(X);
XDataIsAVector = isvector(X);
hasNonSingletonU = ~isempty(find(X.dims == 'U')) && (~XDataIsAVector ...
    || (~hasNoTimeDim && XDataIsAVector && sizeXData(timeDim) == 1) && ~all(sizeXData==1));
if hasNoTimeDim || hasNoChannelDim || hasNonSingletonU
    error(message('deep:dlarray:LstmInvalidFirstInput'))
end

if XHasNoObservationDim
    observationDim = 3;
end

% Retrieve channel and observation dimension index and size
nbChannels = size(X, channelDim);
nbObservations = size(X, observationDim);

nbTimesteps = size(X, timeDim);
if stride < 0
    stride = nbTimesteps;
end

% Order the data for the builtin
% Enforce 'CBT'
dataOrder = [channelDim, observationDim, timeDim];
if ~prod(dataOrder == [1 2 3])
    X = reshape(X, [sizeXData(channelDim) 1 sizeXData(timeDim)]);
else
    X = stripdims(X);
end

% Extract the hidden units and cell states
[h0, h0data] = iExtractState(h0, nbObservations, @iHiddenStateErrorIDs);
[c0, c0data] = iExtractState(c0, nbObservations, @iCellStateErrorIDs);
nbHiddenUnits = size(h0data, 1);
if size(c0data, 1) ~= nbHiddenUnits
    error(message('deep:dlarray:LstmInconsistentStates'));
end

% Validate weights
iValidateWeights(W, 4*nbHiddenUnits, nbChannels);
W = stripdims(W);

% Validate recurrent weights
iValidateRecurrentWeights(R, 4*nbHiddenUnits, nbHiddenUnits);
R = stripdims(R);

% Validate bias
B = iValidateBias(B, 4*nbHiddenUnits);

% Validate LN parameters
a1 = iValidateLayerNormLearnables(a1, 4*nbHiddenUnits);
% b1 = iValidateLayerNormLearnables(b1, 4*nbHiddenUnits);
a2 = iValidateLayerNormLearnables(a2, 4*nbHiddenUnits);
% b2 = iValidateLayerNormLearnables(b2, 4*nbHiddenUnits);
a3 = iValidateLayerNormLearnables(a3, nbHiddenUnits);
b3 = iValidateLayerNormLearnables(b3, nbHiddenUnits);

% Call the internal API
[X, h0data, c0data] = ...
    internal_layer_norm_lstm(X, h0data, c0data, W, R, B, a1, a2, a3, b3, pZoneH, pZoneC, mode);

% Keep required h0data, c0data
h0data = h0data(:,:,stride);
c0data = c0data(:,:,stride);

% Remove the singleton observation dimension if no 'B' dimension in the
% input
if XHasNoObservationDim
    X = squeeze(X);
    X = dlarray(X, 'CT');
    
    h0 = squeeze(h0);
    c0 = squeeze(c0);
else
    X = dlarray(X, 'CBT');
end

% Create the second and third outputs
h0 = iFormatOutputState(h0, h0data);
c0 = iFormatOutputState(c0, c0data);

end

function [Y, HS, CS] = internal_layer_norm_lstm(X, Y0, C0, W, R, b, a1, a2, a3, b3, pZoneH, pZoneC, mode)
%   [X, H, C, Ws] = internal_lstm(X, h0, c0, W, R, b) computes the Long
%   Short Term Memory operation using input data X, initial hidden units
%   h0, initial cell state c0, input weights W, recurrent weights R and
%   bias term b.
%
%   Definitions:
%   nF := Number of features of the input data
%   N := Number of input observations (mini-batch size)
%   nS := Sequence length
%   nH := Hidden units size
%
%   Inputs:
%   X - Input data                  (nF)x(N)x(nS) array
%   W - Input weights               (4*nH)x(nF) matrix
%   R - Recurrent weights           (4*nH)x(nH) matrix
%   b - Bias                        (4*nH)x(1) vector
%   C0 - Initial cell state         (nH)x(N) matrix
%   Y0 - Initial hidden units       (nH)x(N) matrix
%   a1 - Input learnable gain       (4*nH)x(1) vector
%   b1 - Input learnable bias       (4*nH)x(1) vector
%   a2 - Hidden learnable gain     	(4*nH)x(1) vector
%   b2 - Hidden learnable bias     	(4*nH)x(1) vector
%   a3 - Output learnable gain     	(nH)x(1) vector
%   b3 - Output learnable bias     	(nH)x(1) vector
%
%   Outputs:
%   Y - Output                      (nH)x(N)x(nS) array
%   CS - Cell state               	(nH)x(N)x(nS) array
%   HS - Hidden state           	(nH)x(N) matrix

%   Copyright 2019 The MathWorks, Inc.

% Prepare the inputs for cnnhost implementation

returnLast = true; % Always return the full output
recurrentActFn = @internal_sigmoid;
actFn = @tanh;

% Determine dimensions
[F, N, S] = size(X);
H = size(R, 2);

% Pre-allocate output array and cell state
Y = zeros(H, N, S, 'like', X);
CS = zeros(H, N, S, 'like', X);

% Indexing helpers
[zInd, iInd, fInd, oInd] = nnet.internal.cnn.util.gateIndices(H);
ifoInd = [iInd fInd oInd];
epsilon = 1e-5; % For division by zero

% Forward propagate through time
for tt = 1:S
    if tt == 1
        % Layernorm input statistics
        % For input X
        P1 = W*X(:, :, tt);
        mu1 = computeLstmMeans(P1, H, N, zInd, iInd, fInd, oInd);
        sigmaSq1 = computeLstmVars(P1, H, N, zInd, iInd, fInd, oInd);
        % For hidden state Y
        P2 = R*Y0;
        mu2 = computeLstmMeans(P2, H, N, zInd, iInd, fInd, oInd);
        sigmaSq2 = computeLstmVars(P2, H, N, zInd, iInd, fInd, oInd);
        
        % Linear gate operations
%         G = (P1 - mu1)./sqrt(sigmaSq1 + epsilon).*a1 + b1 + ...
%             (P2 - mu2)./sqrt(sigmaSq2 + epsilon).*a2 + b2 + ...
%             b;
        
        G = (P1 - mu1)./sqrt(sigmaSq1 + epsilon).*a1 + ...
            (P2 - mu2)./sqrt(sigmaSq2 + epsilon).*a2 + ...
            b;
        
        % Nonlinear gate operations
        g = iNonlinearActivations( G, zInd, ifoInd, actFn, recurrentActFn );
        
        % Cell state update
        cs = g(zInd, :) .* g(iInd, :) + g(fInd, :) .* C0;
        CS(:, :, 1) = zoneout(pZoneC, F, N, cs, C0, mode);
    else
        % Layernorm input statistics
        % For input X
        P1 = W*X(:, :, tt);
        mu1 = computeLstmMeans(P1, H, N, zInd, iInd, fInd, oInd);
        sigmaSq1 = computeLstmVars(P1, H, N, zInd, iInd, fInd, oInd);
        % For hidden state Y
        P2 = R*Y(:, :, tt - 1);
        mu2 = computeLstmMeans(P2, H, N, zInd, iInd, fInd, oInd);
        sigmaSq2 = computeLstmVars(P2, H, N, zInd, iInd, fInd, oInd);
        
        % Linear gate operations
%         G = (P1 - mu1)./sqrt(sigmaSq1 + epsilon).*a1 + b1 + ...
%             (P2 - mu2)./sqrt(sigmaSq2 + epsilon).*a2 + b2 + ...
%             b;

        G = (P1 - mu1)./sqrt(sigmaSq1 + epsilon).*a1 + ...
            (P2 - mu2)./sqrt(sigmaSq2 + epsilon).*a2 + ...
            b;
        % Nonlinear gate operations
        g = iNonlinearActivations( G, zInd, ifoInd, actFn, recurrentActFn );
        
        % Cell state update
        cs = g(zInd, :) .* g(iInd, :) + g(fInd, :) .* CS(:, :, tt - 1);
        CS(:, :, tt) = zoneout(pZoneC, F, N, cs, CS(:, :, tt - 1), mode);
    end
    % Layernorm output statistics
    mu = mean(CS(:, :, tt), 1);
    sigmaSq = var(CS(:, :, tt), 1, 1);
    
    % Layer output
    y = actFn(...
        (CS(:, :, tt) - mu)./sqrt(sigmaSq + epsilon).*a3 + b3) .* ...
        g(oInd, :);
    
    if tt == 1
        Y(:, :, 1) = zoneout(pZoneH, F, N, y, Y0, mode);
    else
        Y(:, :, tt) = zoneout(pZoneH, F, N, y, Y(:, :, tt - 1), mode);
    end
end

HS = Y;

HS = extractdata(HS);
CS = extractdata(CS);
end

function Y = zoneout(p, F, N, Y, Y_prev, mode)

if strcmp(mode, 'test')
    mask = ones(F, N)*p;
else
    mask = binornd(1, p, F, N);
end

Y = Y_prev.*mask + Y.*(1 - mask);

end

function mu = computeLstmMeans(X, H, N, zInd, iInd, fInd, oInd)
X = extractdata(X);
mu = zeros(4*H,N,'like',X);
idx = {zInd, iInd, fInd, oInd};

for i = 1:4
   mu_i = mean(X(idx{i},:), 1); % Along 'C'
   mu(idx{i},:) = repmat(mu_i,H,1);
end

end

function sigmaSq = computeLstmVars(X, H, N, zInd, iInd, fInd, oInd)
X = extractdata(X);
sigmaSq = zeros(4*H,N,'like',X);
idx = {zInd, iInd, fInd, oInd};

for i = 1:4
   sigmaSq_i = var(X(idx{i},:), 1, 1); % Along 'C'
   sigmaSq(idx{i},:) = repmat(sigmaSq_i,H,1);
end

end

function g = iNonlinearActivations( G, zInd, ifoInd, actFn, recurrentActFn )
% Nonlinear gate operations
g = zeros(size(G), 'like', G);
g(zInd, :) = actFn( G(zInd, :) );
g(ifoInd, :) = recurrentActFn( G(ifoInd, :) );
end

function inputArguments = iParseInputArguments(varargin)
% Parse the optional input arguments

% Default values
defaultFormat = [];

% Create parser
parser = inputParser;
parser.FunctionName = 'lstm';
parser.addParameter('DataFormat', defaultFormat, @(x)validateattributes(x, {'char', 'string'}, {'nonempty'}));

% Parse the arguments
parser.parse(varargin{:});
inputArguments = parser.Results;
end


function [stateInput, data] = iExtractState(stateInput, nObservationsX, errorIDFcn)
% Extract the state input's data, and returns the dlarray shell (if
% any) along with the extracted data

% Extract the data, and get the channel index and observation index.
channelDim = 1; % default for unlabeled
observationDim = 2; % default for unlabeled
if isa(stateInput, 'dlarray')
    if ~isempty(dims(stateInput)) % Labeled dlarray
        % Permute (if needed) to ensure labels are ordered CB
        stateInput = permuteExplicitly(stateInput);
        % Ensure a C dimension was specified
        channelDim = finddatadim(stateInput, 'C');
        if isempty(channelDim)
            error(iGetErrorMessage(errorIDFcn, 'InvalidLabeled'));
        end
        % Find the B dimension
        observationDim = finddatadim(stateInput, 'B');
    end
    % Extract the data
    fd = stateInput.FormattedData;
    [~,data] = extractData(fd);
else % Numeric array
    deep.internal.dlarray.validateNonDlarray(stateInput);
    data = stateInput;
    stateInput = [];
end


% Validate the size of the state input
sizeIn = size(data);
% nbObservationInput = sizeIn(observationIdx);
% observationSizesMismatch = nbObservationInput ~= nObservationsX;
if ~isvector(data) % Input is matrix or N-array
    if isempty(observationDim) % Was specified as a dlarray with no 'B' dimension
        error(iGetErrorMessage(errorIDFcn, 'InvalidLabeled'));
    elseif sizeIn(observationDim) ~= nObservationsX % Has non-singleton 'B' but with an incorrect size
        error(iGetErrorMessage(errorIDFcn, 'InvalidStateObservation'));
    elseif length(sizeIn) > 2 % Too many dimensions
        error(iGetErrorMessage(errorIDFcn, 'TooManyDimensions'));
    end
else
    if sizeIn(channelDim) ~= 1 || isscalar(data) 
        % The vector length is the number of hidden units, need to expand
        % to obtain a CxB matrix
        data = data(:);
        data = data(:, ones(1, nObservationsX));
    else % Only one hidden unit
        % The vector length should be the number of observations
        if length(data) ~= nObservationsX
            error(iGetErrorMessage(errorIDFcn, 'InvalidStateObservation'));
        end
        % Ensure it is a line vector
        data = data(:)';
    end
end
end

function state = iFormatOutputState(state, stateData)
% This function create a dlarray from the stateData and the original
% dlarray (if existing)

if isa(state, 'dlarray')
    fd = state.FormattedData;
    state.FormattedData = [];
    % If the state was specified labeled, change its labels to 'CB'
    if ~isempty(fd.DimensionMetadata)
        fd.DimensionMetadata = 'CB';
    end
    % Insert the data back
    fd = insertData(fd, stateData);
    state.FormattedData = fd;
else % Input was numeric, return the numeric value
    % Unwrap the RecordingArray if needed
    if isa(stateData, 'deep.internal.recording.RecordingArray')
        stateData = stateData.getValue();
    end
    state = stateData;
end

end

function param = iExtractParameter(param, errorIDFcn)
% Extract a param if it is wrapped in a dlarray

if isa(param, 'dlarray')
    fd = param.FormattedData;
    if ~isempty(fd.DimensionMetadata) % Labeled dlarray
        % Perform the internal permutations to obtain X in the builtin order
        param = permuteExplicitly(param);
        fd = param.FormattedData;
        
        % Ensure no 'S', 'T' or 'B' dims
        if (~isempty(finddatadim(param,'S')) || ~isempty(finddatadim(param,'B')) || ~isempty(finddatadim(param,'T')))
            error(iGetErrorMessage(errorIDFcn, 'InvalidLabeled'));
        end
        
        % Make sure there is a C dimension
        channelIdx = finddatadim(param, 'C');
        if isempty(channelIdx)
            error(iGetErrorMessage(errorIDFcn, 'InvalidLabeled'));
        end
        
        % Extract param in the ordered format
        [~,param] = extractData(fd);
        
    else % Unlabeled dlarray
        % Extract weights in the format needed by builtin convolution functions.
        [~,param] = extractData(fd);
    end
else
    deep.internal.dlarray.validateNonDlarray(param);
end

% Validate there are only 2 dimensions
if ~ismatrix(param)
    error(iGetErrorMessage(errorIDFcn, 'TooManyDimensions'));
end
end

function iValidateWeights(weights, expectedSizeFirstDimension, expectedSizeSecondDimension)
% Validate the size of the weights

if size(weights,1) ~= expectedSizeFirstDimension
    error(message('deep:dlarray:LstmWeightsInconsistentFirstDim'))
end
if size(weights,2) ~= expectedSizeSecondDimension
    error(message('deep:dlarray:LstmWeightsInconsistentSecondDim'))
end
end

function iValidateRecurrentWeights(recurrentWeights, expectedSizeFirstDimension, expectedSizeSecondDimension)
% Validate the size of the weights

if size(recurrentWeights,1) ~= expectedSizeFirstDimension
    error(message('deep:dlarray:LstmRecurrentWeightsInconsistentFirstDim'))
end
if size(recurrentWeights,2) ~= expectedSizeSecondDimension
    error(message('deep:dlarray:LstmRecurrentWeightsInconsistentSecondDim'))
end
end

function learnable = iValidateLayerNormLearnables(learnable, expectedSize)
% Validate the size of the weights

if ~isvector(learnable) || (length(learnable) ~= expectedSize)
    error(message('Inconsistent size for a layer normalisation parameter'))
end

% Make column vector
learnable = learnable(:);

end

function bias = iValidateBias(bias, expectedSize)
% Validate the size of the weights

if ~isvector(bias) || (length(bias) ~= expectedSize)
    error(message('deep:dlarray:LstmBiasInconsistentHiddenUnits'))
end

% Make the bias a column vector
bias = bias(:);

end

% Helpers to the throw a more specific error for each input
function msg = iGetErrorMessage(errorIDFcn, IDName)
s = errorIDFcn();
msg =  message(s.(IDName));
end

function s = iHiddenStateErrorIDs()
s = struct(...
    'InvalidStateObservation', 'deep:dlarray:LstmInvalidHiddenStateObservation', ...
    'InvalidLabeled', 'deep:dlarray:LstmInvalidLabeledHiddenState', ...
    'TooManyDimensions', 'deep:dlarray:LstmTooManyDimensionsHiddenState');
end

function s = iCellStateErrorIDs()
s = struct(...
    'InvalidStateObservation', 'deep:dlarray:LstmInvalidCellStateObservation', ...
    'InvalidLabeled', 'deep:dlarray:LstmInvalidLabeledCellState', ...
    'TooManyDimensions', 'deep:dlarray:LstmTooManyDimensionsCellState');
end

function s = iWeightsErrorIDs()
s = struct(...
    'InvalidLabeled', 'deep:dlarray:LstmInvalidLabeledWeights', ...
    'TooManyDimensions', 'deep:dlarray:LstmTooManyDimensionsWeights');
end

function s = iRecurrentWeightsErrorIDs()
s = struct(...
    'InvalidLabeled', 'deep:dlarray:LstmInvalidLabeledRecurrentWeights', ...
    'TooManyDimensions', 'deep:dlarray:LstmTooManyDimensionsRecurrentWeights');
end

function s = iBiasErrorIDs()
s = struct(...
    'InvalidLabeled', 'deep:dlarray:LstmInvalidLabeledBias', ...
    'TooManyDimensions', 'deep:dlarray:LstmTooManyDimensionsBias');
end
