function denoiseBatchTestUD_Banded;
gpuDevice();
addpath('../LASIP');
addpath('../DNCNN/model');
addpath('../DNCNN/utilities');

run /home/cosc/csprh/linux/code/matconvnet-1.0-beta25/matlab/vl_setupnn

validInds = [1 126 72
2 85  2863
3 53  452
4 10  798
5 40  2281
6 17  3200
7 272 784
8 262 1757
9 304 1502
10 314 185
11 253 61
12 546 1679
13 490 388
14 488 3184
15 634 1834
16 452 1107
17 362 270
18 416 2410
19 346 751
20 650 2957
21 726 1757 
22 973 421
23 717 1077
24 724 78
25 743 731
26 855 399
27 737 1532
28 901 1820
29 757 2696
30 764 2410];

scratchDir = '/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI';
baseDir = [scratchDir '/Valid4/'];
modelDir = [scratchDir '/Models/'];



for band = 1:8
epochs = 147;
orig = 1;
nameOfModel = ['out-epoch-' num2str(epochs) '.mat']; 
outPSNRBandsValid4_Orig(band) = multiModelDenoise1(baseDir, nameOfModel,  modelDir, band, validInds(:,3), orig)
end


for band = 1:8
epochs = 120;
orig = 1;
nameOfModel = ['out-epoch-' num2str(epochs) '.mat']; 
outPSNRBandsValid4_Orig120(band) = multiModelDenoise1(baseDir, nameOfModel,  modelDir, band, validInds(:,3), orig)
end

for band = 1:8
epochs = 50;
nameOfModel = ['out-epoch-' num2str(epochs) '.mat']; 
orig = 0;
outPSNRBandsValid4_Fin(band) = multiModelDenoise1(baseDir, nameOfModel,  modelDir, band, validInds(:,3), orig)
end

outPSNRBandsValid4_Fin
outPSNRBandsValid4_Orig

function outPSNRMean = multiModelDenoise1(baseDir, nameOfModel,  modelDir, thisBand, validInds, orig)

load borders;
allIndices = 1:30;
if orig == 0
   modName = ['modelFin_0_' num2str(thisBand) '/' nameOfModel];
else
   modName = ['model_0_' num2str(thisBand) '/' nameOfModel]; 
end
lowEdge = borders(thisBand); highEdge = borders(thisBand+1);
bandIms = allIndices((validInds>=lowEdge)&(validInds<highEdge))
for i = 1 : length(bandIms)
    outPSNR(i) = thisDenoise(bandIms(i), baseDir, modName,  modelDir);
end
outPSNRMean = mean(outPSNR);



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


