function denoiseBatchTestUD_Banded;
gpuDevice();
addpath('../LASIP');
addpath('../DNCNN/model');
addpath('../DNCNN/utilities');

run /home/cosc/csprh/linux/code/matconvnet-1.0-beta25/matlab/vl_setupnn

scratchDir = '/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI';
baseDir = [scratchDir '/Valid/'];
modelDir = [scratchDir '/Models/'];


nameOfModel = 'model_0_0/out-epoch-40.mat'; 
outPSNR40 = singleModelDenoise(baseDir, nameOfModel,  modelDir);
nameOfModel = 'model_0_0/out-epoch-50.mat'; 
outPSNR50 = singleModelDenoise(baseDir, nameOfModel,  modelDir);
nameOfModel = 'model_0_0/out-epoch-60.mat'; 
outPSNR60 = singleModelDenoise(baseDir, nameOfModel,  modelDir);
%nameOfModel = 'model_Huwei_All50-epoch-27.mat'; 
%singleModelDenoise(baseDir, nameOfModel, outDirNN, modelDir);
%nameOfModel = 'model_Huwei_All50a-epoch-60.mat';
%singleModelDenoise(baseDir, nameOfModel, outDirNN, modelDir);
%nameOfModel = 'model_Huwei_All50b-epoch-60.mat';
%singleModelDenoise(baseDir, nameOfModel, outDirNN, modelDir);
nameOfModel = 'out-epoch-60.mat'; 
outPSNR5 = multiModelDenoise1(baseDir, nameOfModel, modelDir);
nameOfModel = 'out-epoch-60.mat'; 
outPSNR6 = multiModelDenoise2(baseDir, nameOfModel, modelDir);
outPSNR5
outPSNR40
outPSNR50
outPSNR60



function outPSNR = multiModelDenoise1(baseDir, nameOfModel,  modelDir)
load testInds;
load borders;

allIndices = 1:30;
for thisBand = 1:8
    modName = ['model_0_' num2str(thisBand) '/' nameOfModel];
    lowEdge = borders(thisBand); highEdge = borders(thisBand+1);
    bandIms = allIndices((testInds>=lowEdge)&(testInds<highEdge))
    for i = 1 : length(bandIms)
        outPSNR(bandIms(i)) = thisDenoise(bandIms(i), baseDir, modName,  modelDir);
    end
end

function outPSNR = multiModelDenoise2(baseDir, nameOfModel, modelDir)

bandIndices{1} = 1:10;
bandIndices{2} = 11:20;
bandIndices{3} = 21:30;

for thisBand = 1:3
    modName = ['model_' num2str(thisBand) '_0/' nameOfModel];
    bandIms = bandIndices{thisBand};
    for i = 1 : length(bandIms)
        outPSNR(bandIms(i)) = thisDenoise(bandIms(i), baseDir, modName,  modelDir);
    end
end

function outPSNR = singleModelDenoise(baseDir, nameOfModel, modelDir)

for i = 1 : 30
    outPSNR(i) = thisDenoise(i, baseDir, nameOfModel, modelDir)
end


function PSNR = thisDenoise(ImNum, baseDir, nameOfModel,  modelDir)

load(fullfile(modelDir, nameOfModel));

net = vl_simplenn_tidy(net);
net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);

noisyName = [baseDir 'Noisy/Test_Image_' num2str(ImNum) '.png'];
cleanName = [baseDir 'Clean/Test_Image_' num2str(ImNum) '.png'];

zRGB = im2double(imread(noisyName));
yRGB = im2double(imread(cleanName));
input = single(zRGB);
clean = single(yRGB);

res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test','CuDNN', 'true');

output = input - res(end).x;

[PSNR, SSIM] = Cal_PSNRSSIM(255*clean,255*output,0,0);


