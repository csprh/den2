
%%% This is the testing code demo for color image (Gaussian) denoising.
%%% The model is trained with 1) noise levels in [0 55]; 2) 432 training images.


% clear; clc;
addpath('/Users/csprh/Dlaptop/MATLAB/MYCODE/DENOISE/den2/bm3d');

addpath('utilities');
%folderTest  = '/space/csprh/WATER/material20190426/CLEAR/PNGs';
%folderTestOut = '/space/csprh/WATER/material20190426/CLEAR/Out'
folderTest  = '/Users/csprh/Downloads/PNGs/PNGs'; %%% test dataset
folderTestOut = '/Users/csprh/Downloads/PNGs/Out2'
folderModel = 'model'; 
noiseSigma  = 45;  %%% image noise level
showResult  = 1;
useGPU      = 0;
pauseTime   = 1;



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
    
    label = im2double(label);
    
    [PSNR, yRGB_est] = CBM3D(label, label, 10);
    imwrite(yRGB_est, fullfile(folderTestOut,filePaths(i).name));
    
end

disp(mean(PSNRs));


