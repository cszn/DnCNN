function res = vl_simplenn(net, x, dzdy, res, varargin)
%VL_SIMPLENN  Evaluate a SimpleNN network.
%   RES = VL_SIMPLENN(NET, X) evaluates the convnet NET on data X.
%   RES = VL_SIMPLENN(NET, X, DZDY) evaluates the convnent NET and its
%   derivative on data X and output derivative DZDY (foward+bacwkard pass).
%   RES = VL_SIMPLENN(NET, X, [], RES) evaluates the NET on X reusing the
%   structure RES.
%   RES = VL_SIMPLENN(NET, X, DZDY, RES) evaluates the NET on X and its
%   derivatives reusing the structure RES.
%
%   This function process networks using the SimpleNN wrapper
%   format. Such networks are 'simple' in the sense that they consist
%   of a linear sequence of computational layers. You can use the
%   `dagnn.DagNN` wrapper for more complex topologies, or write your
%   own wrapper around MatConvNet computational blocks for even
%   greater flexibility.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
global sigmas;
opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false ;
opts.parameterServer = [] ;
opts.holdOn = false ;
opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;
assert(opts.backPropDepth > 0, 'Invalid `backPropDepth` value (!>0)');
backPropLim = max(n - opts.backPropDepth + 1, 1);

if (nargin <= 2) || isempty(dzdy)
    doder = false ;
    if opts.skipForward
        error('simplenn:skipForwardNoBackwPass', ...
            '`skipForward` valid only when backward pass is computed.');
    end
else
    doder = true ;
end

if opts.cudnn
    cudnn = {'CuDNN'} ;
    bnormCudnn = {'NoCuDNN'} ; % ours seems slighty faster
else
    cudnn = {'NoCuDNN'} ;
    bnormCudnn = {'NoCuDNN'} ;
end

switch lower(opts.mode)
    case 'normal'
        testMode = false ;
    case 'test'
        testMode = true ;
    otherwise
        error('Unknown mode ''%s''.', opts. mode) ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
    if opts.skipForward
        error('simplenn:skipForwardEmptyRes', ...
            'RES structure must be provided for `skipForward`.');
    end
    res = struct(...
        'x', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ...
        'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), ...
        'stats', cell(1,n+1), ...
        'time', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1))) ;
end

if ~opts.skipForward
    res(1).x = x ;
end

% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
    if opts.skipForward, break; end;
    l = net.layers{i} ;
    %res(i).time = tic ;
    switch l.type
        case 'conv'
            res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                'pad', l.pad, ...
                'stride', l.stride, ...
                'dilate', l.dilate, ...
                l.opts{:}, ...
                cudnn{:}) ;
            
        case 'concat'
            if size(sigmas,1)~=size(res(i).x,1)
                sigmaMap   = bsxfun(@times,ones(size(res(i).x,1),size(res(i).x,2),1,size(res(i).x,4)),permute(sigmas,[3 4 1 2])) ;
                res(i+1).x = vl_nnconcat({res(i).x,sigmaMap}) ;
            else
                res(i+1).x = vl_nnconcat({res(i).x,sigmas}) ;
            end
            
        case 'SubP'
            res(i+1).x = vl_nnSubP(res(i).x, [],'scale',l.scale) ;
        case 'relu'
            leak = {} ; 
            res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;
    end
    
    % optionally forget intermediate results
    needsBProp = doder && i >= backPropLim;
    forget = opts.conserveMemory && ~needsBProp ;
    if i > 1
        lp = net.layers{i-1} ;
        % forget RELU input, even for BPROP
        forget = forget && (~needsBProp || (strcmp(l.type, 'relu') && ~lp.precious)) ;
        forget = forget && ~(strcmp(lp.type, 'loss') || strcmp(lp.type, 'softmaxloss')) ;
        forget = forget && ~lp.precious ;
    end
    if forget
        res(i).x = [] ;
    end
    
    if gpuMode && opts.sync
        wait(gpuDevice) ;
    end
    %res(i).time = toc(res(i).time) ;
end
