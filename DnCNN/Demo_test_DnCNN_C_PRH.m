
%%% This is the testing code demo for color image (Gaussian) denoising.
%%% The model is trained with 1) noise levels in [0 55]; 2) 432 training images.

%run /Users/csprh/Dlaptop/MATLAB/TOOLBOXES/matconvnet-1.0-beta25/matlab/vl_setupnn
run /mnt/storage/home/csprh/code/matconvnet-1.0-beta25i/matlab/vl_setupnn
% clear; clc;
%addpath('utilities');
folderTest  = '/space/csprh/WATER/material20190426/CLEAR/PNGs';
folderTestOut = '/space/csprh/WATER/material20190426/CLEAR/Out'
%folderTest  = '/Users/csprh/Downloads/PNGs/PNGs'; %%% test dataset
%folderTestOut = '/Users/csprh/Downloads/PNGs/Out'
folderModel = 'model'; 
noiseSigma  = 45;  %%% image noise level
showResult  = 1;
useGPU      = 0;
pauseTime   = 1;


%%% load blind Gaussian denoising model (color image)
load(fullfile(folderModel,'GD_Color_Blind.mat')); %%% for sigma in [0,55]
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
    
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    %res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.
    output = input - res(end).x;
    output(output>1) = 1;
    output(output<0) = 0;
    imwrite(fullfile(folderTestOut,filePaths(i).name),output);
    
end

disp(mean(PSNRs));


