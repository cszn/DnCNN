
%%% Note: run the 'GenerateTrainingPatches.m' to generate
%%% training data (clean images) first.

rng('default')

global sigma; %%% noise level
sigma = 25;

%%%-------------------------------------------------------------------------
%%% Configuration
%%%-------------------------------------------------------------------------
opts.modelName        = 'model_64_25_Res_Bnorm_Adam'; %%% model name
opts.learningRate     = [logspace(-3,-3,30) logspace(-4,-4,20)];%%% you can change the learning rate
opts.batchSize        = 128;
opts.gpus             = [1]; %%% this code can only support one GPU!
opts.numSubBatches    = 2;
opts.bnormLearningRate= 0;
           
%%% solver
opts.solver           = 'Adam';
opts.numberImdb       = 1;

opts.imdbDir          = 'data/TrainingPatches/imdb_40_128.mat';

opts.gradientClipping = false; %%% set 'true' to prevent exploding gradients in the beginning.
opts.backPropDepth    = Inf;
%%%------------;-------------------------------------------------------------
%%%   Initialize model and load data
%%%-------------------------------------------------------------------------
%%%  model
net  = feval(['DnCNN_init_',opts.modelName]);

%%%  load data
opts.expDir      = fullfile('data', opts.modelName);

%%%-------------------------------------------------------------------------
%%%   Train 
%%%-------------------------------------------------------------------------

[net, info] = DnCNN_train(net,  ...
    'expDir', opts.expDir, ...
    'learningRate',opts.learningRate, ...
    'bnormLearningRate',opts.bnormLearningRate, ...
    'numSubBatches',opts.numSubBatches, ...
    'numberImdb',opts.numberImdb, ...
    'backPropDepth',opts.backPropDepth, ...
    'imdbDir',opts.imdbDir, ...
    'solver',opts.solver, ...
    'gradientClipping',opts.gradientClipping, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;






