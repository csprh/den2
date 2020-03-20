
%%% This is the testing code demo for color image (Gaussian) denoising.
%%% The model is trained with 1) noise levels in [0 55]; 2) 432 training images.

%run /Users/csprh/Dlaptop/MATLAB/TOOLBOXES/matconvnet-1.0-beta25/matlab/vl_setupnn
%run /home/cosc/csprh/linux/code/matconvnet/matlab/vl_setupnn
run /mnt/storage/home/csprh/code/matconvnet-1.0-beta25/matlab/vl_setupnn
% clear; clc;
addpath('utilities');
%folderTest  = '/space/csprh/ICME/PNGS/S1ElFuente-Palacio_1920x1080_60fps_10bit_420';
%folderTestOut = '/space/csprh/ICME/DENOISED1/S1ElFuente-Palacio_1920x1080_60fps_10bit_420'
%folderTestOut = '/space/csprh/ICME/DENOISED1/S1ElFuente-Palacio_1920x1080_60fps_10bit_420'
folderTest  = '/mnt/storage/scratch/csprh/ICME/PNGS/S1ElFuente-Palacio_1920x1080_60fps_10bit_420';
folderTestOut  = '/mnt/storage/scratch/csprh/ICME/DENOISED2/S1ElFuente-Palacio_1920x1080_60fps_10bit_420';

%folderTest  = '/Volumes/David_Bull/ICME_GC/ORIG/S1ElFuente-Palacio_1920x1080_60fps_10bit_420'; %%% test dataset
%folderTestOut = '/Volumes/David_Bull/ICME_GC/DENOISED1/S1ElFuente-Palacio_1920x1080_60fps_10bit_420';
folderModel = 'model'; 
noiseSigma  = 45;  %%% image noise level
showResult  = 1;
useGPU      = 0;
pauseTime   = 1;


%%% load blind Gaussian denoising model (color image)
%load(fullfile(folderModel,'GD_Color_Blind.mat')); %%% for sigma in [0,55]
%load(fullfile(folderModel,'model_Huwei_All50b-epoch-60.mat'));
load(fullfile('/mnt/storage/home/csprh/code/HUAWEI/DNCNN/TrainingCodes/DnCNNHuawei/data/model_Huwei_All/model_Huwei_All-epoch-19.mat'));
net = vl_simplenn_tidy(net);
%%%
% net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

%%% move to gpu
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%%% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

%%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));

for i = 1:length(filePaths)
    
    %%% read current image
    label = imread(fullfile(folderTest,filePaths(i).name));
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    label = im2double(label);
    
    %%% add Gaussian noise
    randn('seed',0);
    input = single(label);
    
    %%% convert to GPU
    if useGPU
        input = gpuArray(input);
    end
    
    %res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.
    output = input - res(end).x;
    output(output>1) = 1;
    output(output<0) = 0;
    imwrite(output, fullfile(folderTestOut,filePaths(i).name));
    
end

disp(mean(PSNRs));


