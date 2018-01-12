function net = DnCNN_Init()

% by Kai Zhang (1/2018)
% cskaizhang@gmail.com
% https://github.com/cszn

% Create DAGNN object
net = dagnn.DagNN();

% conv + relu
blockNum = 1;
inVar = 'input';
channel= 1; % grayscale image
dims   = [3,3,channel,64];
pad    = [1,1];
stride = [1,1];
lr     = [1,1];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

for i = 1:15
    % conv + bn + relu
    dims   = [3,3,64,64];
    pad    = [1,1];
    stride = [1,1];
    lr     = [1,0];
    [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
    n_ch   = dims(4);
    [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch);
    [net, inVar, blockNum] = addReLU(net, blockNum, inVar);
end

% conv
dims   = [3,3,64,channel];
pad    = [1,1];
stride = [1,1];
lr     = [1,0]; % or [1,1], it does not influence the results
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);

% sum
inVar = {inVar,'input'};
[net, inVar, blockNum] = addSum(net, blockNum, inVar);

outputName = 'prediction';
net.renameVar(inVar,outputName)

% loss
net.addLayer('loss', dagnn.Loss('loss','L2'), {'prediction','label'}, {'objective'},{});
net.vars(net.getVarIndex('prediction')).precious = 1;


end




% Add a Concat layer
function [net, inVar, blockNum] = addConcat(net, blockNum, inVar)

outVar   = sprintf('concat%d', blockNum);
layerCur = sprintf('concat%d', blockNum);

block = dagnn.Concat('dim',3);
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end


% Add a loss layer
function [net, inVar, blockNum] = addLoss(net, blockNum, inVar)

outVar   = 'objective';
layerCur = sprintf('loss%d', blockNum);

block    = dagnn.Loss('loss','L2');
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a sum layer
function [net, inVar, blockNum] = addSum(net, blockNum, inVar)

outVar   = sprintf('sum%d', blockNum);
layerCur = sprintf('sum%d', blockNum);

block    = dagnn.Sum();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a relu layer
function [net, inVar, blockNum] = addReLU(net, blockNum, inVar)

outVar   = sprintf('relu%d', blockNum);
layerCur = sprintf('relu%d', blockNum);

block    = dagnn.ReLU('leak',0);
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a bnorm layer
function [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch)

trainMethod = 'adam';
outVar   = sprintf('bnorm%d', blockNum);
layerCur = sprintf('bnorm%d', blockNum);

params={[layerCur '_g'], [layerCur '_b'], [layerCur '_m']};
net.addLayer(layerCur, dagnn.BatchNorm('numChannels', n_ch), {inVar}, {outVar},params) ;

pidx = net.getParamIndex({[layerCur '_g'], [layerCur '_b'], [layerCur '_m']});
b_min                           = 0.025;
net.params(pidx(1)).value       = clipping(sqrt(2/(9*n_ch))*randn(n_ch,1,'single'),b_min);
net.params(pidx(1)).learningRate= 1;
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(1)).trainMethod = trainMethod;

net.params(pidx(2)).value       = zeros(n_ch, 1, 'single');
net.params(pidx(2)).learningRate= 1;
net.params(pidx(2)).weightDecay = 0;
net.params(pidx(2)).trainMethod = trainMethod;

net.params(pidx(3)).value       = [zeros(n_ch,1,'single'), 0.01*ones(n_ch,1,'single')];
net.params(pidx(3)).learningRate= 1;
net.params(pidx(3)).weightDecay = 0;
net.params(pidx(3)).trainMethod = 'average';

inVar    = outVar;
blockNum = blockNum + 1;
end


% add a ConvTranspose layer
function [net, inVar, blockNum] = addConvt(net, blockNum, inVar, dims, crop, upsample, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('convt%d', blockNum);

layerCur    = sprintf('convt%d', blockNum);

convBlock = dagnn.ConvTranspose('size', dims, 'crop', crop,'upsample', upsample, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f  = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*dims(4))) ; %improved Xavier
net.params(f).value        = sc*randn(dims, 'single');
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(3), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end


% add a Conv layer
function [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts    = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('conv%d', blockNum);
layerCur    = sprintf('conv%d', blockNum);

convBlock   = dagnn.Conv('size', dims, 'pad', pad,'stride', stride, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*max(dims(3), dims(4)))) ; %improved Xavier
net.params(f).value        = sc*randn(dims, 'single') ;
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(4), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end


function A = clipping(A,b)
A(A>=0&A<b) = b;
A(A<0&A>-b) = -b;
end
