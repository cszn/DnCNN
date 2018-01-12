function [net,stats] = DnCNN_train_dag(net, varargin)

%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


%%%-------------------------------------------------------------------------
%%% solvers: SGD(default) and Adam with(default)/without gradientClipping
%%%-------------------------------------------------------------------------

%%% solver: Adam
%%% opts.solver = 'Adam';
opts.beta1   = 0.9;
opts.beta2   = 0.999;
opts.alpha   = 0.01;
opts.epsilon = 1e-8;

%%% solver: SGD
opts.solver = 'SGD';
opts.learningRate = 0.01;
opts.weightDecay  = 0.0005;
opts.momentum     = 0.9 ;

%%% GradientClipping
opts.gradientClipping = false;
opts.theta            = 0.005;

%%%-------------------------------------------------------------------------
%%%  setting for dag
%%%-------------------------------------------------------------------------

opts.conserveMemory = true;
opts.mode = 'normal';
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;
opts.numSubBatches = 1;
%%%-------------------------------------------------------------------------
%%%  setting for model
%%%-------------------------------------------------------------------------

opts.batchSize = 128 ;
opts.gpus = [];
opts.numEpochs = 300 ;
opts.modelName   = 'model';
opts.expDir = fullfile('data',opts.modelName) ;
opts.numberImdb   = 1;
opts.imdbDir      = opts.expDir;
opts.derOutputs = {'objective', 1} ;

%%%-------------------------------------------------------------------------
%%%  update settings
%%%-------------------------------------------------------------------------

opts = vl_argparse(opts, varargin);
opts.numEpochs = numel(opts.learningRate);

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

%%% load training data
% opts.imdbPath    = fullfile(opts.imdbDir, 'imdb.mat');
% imdb = load(opts.imdbPath) ;

% if  mod(epoch,5)~=1 && isfield(imdb,'set') ~= 0
%
% else
%     clear imdb;
%     [imdb] = generatepatches;
% end
%
% opts.train = find(imdb.set==1);

opts.continue = true;
opts.prefetch = true;
opts.saveMomentum = false;
opts.nesterovUpdate = false ;

opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;


opts.extractStatsFn = @extractStats ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------
opts.train = true;
evaluateMode = isempty(opts.train) ;

% -------------------------------------------------------------------------
%                                                        Train
% -------------------------------------------------------------------------


modelPath = @(ep) fullfile(opts.expDir, sprintf([opts.modelName,'-epoch-%d.mat'], ep));
start = findLastCheckpoint(opts.expDir,opts.modelName) ;

if start>=1
    fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
    [net] = loadState(modelPath(start)) ;
end
state = [] ;


% for iobj = numel(opts.derOutputs)
net.vars(net.getVarIndex(opts.derOutputs{1})).precious = 1;
% end

imdb = [];

for epoch=start+1:opts.numEpochs
    
    if  mod(epoch,10)~=1 && isfield(imdb,'set') ~= 0
        
    else
        clear imdb;
        [imdb] = generatepatches;
    end
    
    opts.train = find(imdb.set==1);
    
    prepareGPUs(opts, epoch == start+1) ;
    
    % Train for one epoch.
    params = opts;
    params.epoch = epoch ;
    params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    params.thetaCurrent = opts.theta(min(epoch, numel(opts.theta)));
    params.train = opts.train(randperm(numel(opts.train))) ; % shuffle
    params.getBatch = getBatch ;
    
    if numel(opts.gpus) <= 1
        [net,~] = processEpoch(net, state, params, 'train',imdb) ;
        if ~evaluateMode
            saveState(modelPath(epoch), net) ;
        end
        %  lastStats = state.stats ;
    else
        spmd
            [net, ~] = processEpoch(net, state, params, 'train',imdb) ;
            if labindex == 1 && ~evaluateMode
                saveState(modelPath(epoch), net) ;
            end
            %  lastStats = state.stats ;
        end
        %lastStats = accumulateStats(lastStats) ;
    end
    
    % stats.train(epoch) = lastStats.train ;
    % stats.val(epoch) = lastStats.val ;
    % clear lastStats ;
    % saveStats(modelPath(epoch), stats) ;
    
end

% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1} ; end

% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, state, params, mode, imdb)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.
% initialize with momentum 0
if isempty(state) || isempty(state.momentum)
    state.momentum = num2cell(zeros(1, numel(net.params))) ;
    state.m = num2cell(zeros(1, numel(net.params))) ;
    state.v = num2cell(zeros(1, numel(net.params))) ;
    state.t = num2cell(zeros(1, numel(net.params))) ;
end

% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
    net.move('gpu') ;
    state.momentum = cellfun(@gpuArray, state.momentum, 'uniformoutput', false) ;
    state.m = cellfun(@gpuArray,state.m,'UniformOutput',false) ;
    state.v = cellfun(@gpuArray,state.v,'UniformOutput',false) ;
    state.t = cellfun(@gpuArray,state.t,'UniformOutput',false) ;
end


if numGpus > 1
    parserv = ParameterServer(params.parameterServer) ;
    net.setParameterServer(parserv) ;
else
    parserv = [] ;
end

% profile
if params.profile
    if numGpus <= 1
        profile clear ;
        profile on ;
    else
        mpiprofile reset ;
        mpiprofile on ;
    end
end

num = 0 ;
epoch = params.epoch ;
subset = params.(mode) ;
%adjustTime = 0 ;

stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
count = 0;
%start = tic ;
for t=1:params.batchSize:numel(subset)
    %     fprintf('%s: epoch %02d: %3d/%3d:', mode, epoch, ...
    %         fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
    
    batchSize = min(params.batchSize, numel(subset) - t + 1) ;
    count = count + 1;
    for s=1:params.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+params.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        
        inputs = params.getBatch(imdb, batch) ;
        
        if params.prefetch
            if s == params.numSubBatches
                batchStart = t + (labindex-1) + params.batchSize ;
                batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
            params.getBatch(imdb, nextBatch) ;
        end
        
        if strcmp(mode, 'train')
            net.mode = 'normal' ;
            net.accumulateParamDers = (s ~= 1) ;
            net.eval(inputs, params.derOutputs, 'holdOn', s < params.numSubBatches) ;
        else
            net.mode = 'test' ;
            net.eval(inputs) ;
        end
    end
    
    % Accumulate gradient.
    if strcmp(mode, 'train')
        if ~isempty(parserv), parserv.sync() ; end
        state = accumulateGradients(net, state, params, batchSize, parserv) ;
    end
    
    
    
    %%%--------add your code here------------------------
    
    
    %%%--------------------------------------------------
    loss2 = squeeze(gather(net.vars(net.getVarIndex(params.derOutputs{1})).value));
   
    fprintf('%s: epoch %02d : %3d/%3d:', mode, epoch,  ...
        fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
    fprintf('error: %f \n', loss2) ;
    
end

% Save back to state.
state.stats.(mode) = stats ;
if params.profile
    if numGpus <= 1
        state.prof.(mode) = profile('info') ;
        profile off ;
    else
        state.prof.(mode) = mpiprofile('info');
        mpiprofile off ;
    end
end
if ~params.saveMomentum
    state.momentum = [] ;
    state.m = [] ;
    state.v = [] ;
    state.t = [] ;
else
    state.momentum = cellfun(@gather, state.momentum, 'uniformoutput', false) ;
    state.m = cellfun(@gather, state.m, 'uniformoutput', false) ;
    state.v = cellfun(@gather, state.v, 'uniformoutput', false) ;
    state.t = cellfun(@gather, state.t, 'uniformoutput', false) ;
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
% numGpus = numel(params.gpus) ;
% otherGpus = setdiff(1:numGpus, labindex) ;
for p=1:numel(net.params)
    
    if ~isempty(parserv)
        parDer = parserv.pullWithIndex(p) ;
    else
        parDer = net.params(p).der ;
    end
    
    switch params.solver
        
        case 'SGD' %%% solver: SGD
            
            switch net.params(p).trainMethod
                
                case 'average' % mainly for batch normalization
                    thisLR = net.params(p).learningRate;
                    net.params(p).value = vl_taccum(...
                        1 - thisLR, net.params(p).value, ...
                        (thisLR/batchSize/net.params(p).fanout),  parDer) ;
                    
                otherwise
                    thisDecay = params.weightDecay * net.params(p).weightDecay ;
                    thisLR = params.learningRate * net.params(p).learningRate ;
                    
                    % Normalize gradient and incorporate weight decay.
                    parDer = vl_taccum(1/batchSize, parDer, ...
                        thisDecay, net.params(p).value) ;
                    
                    theta  = params.thetaCurrent/lr;
                    parDer = gradientClipping(parDer,theta,params.gradientClipping);
                    
                    % Update momentum.
                    state.momentum{p} = vl_taccum(...
                        params.momentum, state.momentum{p}, ...
                        -1, parDer) ;
                    
                    % Nesterov update (aka one step ahead).
                    if params.nesterovUpdate
                        delta = vl_taccum(...
                            params.momentum, state.momentum{p}, ...
                            -1, parDer) ;
                    else
                        delta = state.momentum{p} ;
                    end
                    
                    % Update parameters.
                    net.params(p).value = vl_taccum(...
                        1,  net.params(p).value, thisLR, delta) ;
            end
            
        case  'Adam'
            
            switch net.params(p).trainMethod
                
                case 'average' % mainly for batch normalization
                    thisLR = net.params(p).learningRate;
                    net.params(p).value = vl_taccum(...
                        1 - thisLR, net.params(p).value, ...
                        (thisLR/batchSize/net.params(p).fanout),  parDer) ;

                otherwise
                    
                    thisLR = params.learningRate * net.params(p).learningRate ;
                    state.t{p} = state.t{p} + 1;
                    t = state.t{p};
                    alpha = thisLR; % opts.alpha; 
                    lr = alpha * sqrt(1 - params.beta2^t) / (1 - params.beta1^t);
                    
                    state.m{p} = state.m{p} + (1 - params.beta1) .* (net.params(p).der - state.m{p});
                    state.v{p} = state.v{p} + (1 - params.beta2) .* (net.params(p).der .* net.params(p).der - state.v{p});
                    net.params(p).value = net.params(p).value - lr * state.m{p} ./ (sqrt(state.v{p}) + params.epsilon);%   - thisLR * 0.0005 * net.params(p).value;
            end
    end
end


%%%-------------------------------------------------------------------------
function A = smallClipping(A, theta)
%%%-------------------------------------------------------------------------
A(A>theta)  = A(A>theta) -0.0001;
A(A<-theta) = A(A<-theta)+0.0001;
%%%-------------------------------------------------------------------------
function A = smallClipping2(A, theta1,theta2)
%%%-------------------------------------------------------------------------
A(A>theta1)  = A(A>theta1)-0.02;
A(A<theta2)  = A(A<theta2)+0.02;

function A = smallClipping3(A, theta1,theta2)
%%%-------------------------------------------------------------------------
A(A>theta1)  = A(A>theta1) -0.1;
A(A<theta2)  = A(A<theta2) +0.1;
% % -------------------------------------------------------------------------
% function stats = accumulateStats(stats_)
% % -------------------------------------------------------------------------
%
% for s = {'train', 'val'}
%     s = char(s) ;
%     total = 0 ;
%
%     % initialize stats stucture with same fields and same order as
%     % stats_{1}
%     stats__ = stats_{1} ;
%     names = fieldnames(stats__.(s))' ;
%     values = zeros(1, numel(names)) ;
%     fields = cat(1, names, num2cell(values)) ;
%     stats.(s) = struct(fields{:}) ;
%
%     for g = 1:numel(stats_)
%         stats__ = stats_{g} ;
%         num__ = stats__.(s).num ;
%         total = total + num__ ;
%
%         for f = setdiff(fieldnames(stats__.(s))', 'num')
%             f = char(f) ;
%             stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;
%
%             if g == numel(stats_)
%                 stats.(s).(f) = stats.(s).(f) / total ;
%             end
%         end
%     end
%     stats.(s).num = total ;
% end

% -------------------------------------------------------------------------
function stats = extractStats(stats, net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
for i = 1:numel(sel)
    stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net_)
% -------------------------------------------------------------------------
net = net_.saveobj() ;
save(fileName, 'net') ;

% -------------------------------------------------------------------------
function saveStats(fileName, stats)
% -------------------------------------------------------------------------
if exist(fileName)
    save(fileName, 'stats', '-append') ;
else
    save(fileName, 'stats') ;
end

% -------------------------------------------------------------------------
function [net] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net') ;
net = dagnn.DagNN.loadobj(net) ;
% if isempty(whos('stats'))
%     error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
%         fileName) ;
% end

%%%-------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir,modelName)
%%%-------------------------------------------------------------------------
list = dir(fullfile(modelDir, [modelName,'-epoch-*.mat'])) ;
tokens = regexp({list.name}, [modelName,'-epoch-([\d]+).mat'], 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;


% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
    % check parallel pool integrity as it could have timed out
    pool = gcp('nocreate') ;
    if ~isempty(pool) && pool.NumWorkers ~= numGpus
        delete(pool) ;
    end
    pool = gcp('nocreate') ;
    if isempty(pool)
        parpool('local', numGpus) ;
        cold = true ;
    end
    
end
if numGpus >= 1 && cold
    fprintf('%s: resetting GPU\n', mfilename)
    clearMex() ;
    if numGpus == 1
        gpuDevice(opts.gpus)
    else
        spmd
            clearMex() ;
            gpuDevice(opts.gpus(labindex))
        end
    end
end


%%%-------------------------------------------------------------------------
function A = gradientClipping(A, theta,gradientClip)
%%%-------------------------------------------------------------------------
if gradientClip
    A(A>theta)  = theta;
    A(A<-theta) = -theta;
else
    return;
end

% -------------------------------------------------------------------------
function fn = getBatch()
% -------------------------------------------------------------------------
fn = @(x,y) getDagNNBatch(x,y) ;

% -------------------------------------------------------------------------
function [inputs2] = getDagNNBatch(imdb, batch)
% -------------------------------------------------------------------------
noiselevel = 25;
label      = imdb.labels(:,:,:,batch);
label      = data_augmentation(label,randi(8));
input      = label + noiselevel/255*randn(size(label),'single');  % add AWGN with noise level noiselevel
input      = gpuArray(input);
label      = gpuArray(label);
inputs2    = {'input', input, 'label', label} ;


