
%%% Note: run the 'GenerateTrainingPatches.m' to generate
%%% training data (clean images) first.

rng('default')

global sigma; %%% noise level
sigma = 25;

run /home/cosc/csprh/linux/code/matconvnet-1.0-beta25/matlab/vl_setupnn
%%%-------------------------------------------------------------------------
%%% Configuration
%%%-------------------------------------------------------------------------
opts.modelName        = 'model_InSAR_1_0_NOGRAD'; %%% model name
opts.learningRate     = [logspace(-3,-3,30) logspace(-4,-4,20) logspace(-5,-5,20)];%%% you can change the learning rate
%opts.learningRate     = [logspace(-3,-3,12) logspace(-4,-4,7)];
opts.batchSize        = 128;
opts.gpus             = [1]; %%% this code can only support one GPU!
opts.numSubBatches    = 2;
opts.bnormLearningRate= 0;
           
%%% solver
opts.solver           = 'Adam';
opts.numberImdb       = 1;

opts.imdbDir          = 'data/TrainingPatches/imdb_40_128_NOGRAD.mat';

opts.gradientClipping = false; %%% set 'true' to prevent exploding gradients in the beginning.
opts.backPropDepth    = Inf;
%%%------------;-------------------------------------------------------------
%%%   Initialize model and load data
%%%-------------------------------------------------------------------------
%%%  model
net  = feval('DnCNN_init_model_InSAR_NOGRAD');

%%%  load data
opts.expDir      = fullfile('data', opts.modelName);

%%%-------------------------------------------------------------------------
%%%   Train 
%%%-------------------------------------------------------------------------

[net, info] = DnCNN_train_NOGRAD(net,  ...
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






