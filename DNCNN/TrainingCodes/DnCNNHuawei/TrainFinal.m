function TrainUD_AllModels
run /home/cosc/csprh/linux/code/matconvnet-1.0-beta25/matlab/vl_setupnn


for theseBands = 1:8
     createModels(0,theseBands);
end


function createModels(theseClasses,theseBands)
rng('default')

dataBase      = '/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI/TrainingPatches';
modelBase      = '/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI/Models';
thisModel = fullfile(dataBase,['imdb_',num2str(theseClasses),'_',num2str(theseBands)]);
modelName = fullfile(modelBase,['model_',num2str(theseClasses),'_',num2str(theseBands)]);
%%%-----------------------------------------------------------i--------------
%%% Configuration
%%%-------------------------------------------------------------------------
opts.modelName        = modelName; %%% model name
opts.learningRate     = [logspace(-3,-3,30) logspace(-4,-4,20)];%%% you can change the learning rate
opts.learningRate     = [logspace(-3,-3,30) logspace(-3,-5,30) logspace(-3,-4,30) logspace(-3,-5,30) logspace(-4.9,-5.1,30)];
%opts.learningRate     = [logspace(-3,-3,12) logspace(-4,-4,7)];
opts.batchSize        = 128;
opts.gpus             = [1]; %%% this code can only support one GPU!
opts.numSubBatches    = 2;
opts.bnormLearningRate= 0;
           
%%% solver
opts.solver           = 'Adam';
opts.numberImdb       = 1;

opts.imdbDir          = thisModel;

opts.gradientClipping = false; %%% set 'true' to prevent exploding gradients in the beginning.
opts.backPropDepth    = Inf;
%%%------------;-------------------------------------------------------------
%%%   Initialize model and load data
%%%-------------------------------------------------------------------------
%%%  model
net  = feval('DnCNN_init_model_Huwei_All');

%%%  load data
opts.expDir      = opts.modelName;
opts.modelName = 'out';
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






