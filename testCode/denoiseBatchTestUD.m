function denoiseBatchTestUD_Banded;
gpuDevice();
addpath('../LASIP');
addpath('../DNCNN/model');
addpath('../DNCNN/utilities');

run /home/cosc/csprh/linux/code/matconvnet-1.0-beta25/matlab/vl_setupnn

scratchDir = '/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI';
baseDir = [scratchDir '/Test/'];
modelDir = [scratchDir '/Models/'];


%nameOfModel = 'model_0_0/out-epoch-60.mat'; outDirNN = 'OutUoBCNN_1/';
%singleModelDenoise(baseDir, nameOfModel, outDirNN, modelDir);
%nameOfModel = 'model_Huwei_All50-epoch-27.mat'; outDirNN = 'OutUoBCNN_2/';
%singleModelDenoise(baseDir, nameOfModel, outDirNN, modelDir);
%nameOfModel = 'model_Huwei_All50a-epoch-60.mat'; outDirNN = 'OutUoBCNN_3/';
%singleModelDenoise(baseDir, nameOfModel, outDirNN, modelDir);
%nameOfModel = 'model_Huwei_All50b-epoch-60.mat'; outDirNN = 'OutUoBCNN_4/';
%singleModelDenoise(baseDir, nameOfModel, outDirNN, modelDir);
%nameOfModel = 'out-epoch-60.mat'; outDirNN = 'OutUoBCNN_5/';
%multiModelDenoise1(baseDir, nameOfModel, outDirNN, modelDir);
%nameOfModel = 'out-epoch-60.mat'; outDirNN = 'OutUoBCNN_6/';
%multiModelDenoise2(baseDir, nameOfModel, outDirNN, modelDir);
nameOfModel = 'out-epoch-60.mat'; outDirNN = 'OutUoBCNN_7/';
multiModelDenoise3(baseDir, nameOfModel, outDirNN, modelDir)

function multiModelDenoise1(baseDir, nameOfModel, outDirNN, modelDir)
load testInds;
load borders;

allIndices = 1:30;
for thisBand = 1:8
    modName = ['model_0_' num2str(thisBand) '/' nameOfModel];
    lowEdge = borders(thisBand); highEdge = borders(thisBand+1);
    bandIms = allIndices((testInds>=lowEdge)&(testInds<highEdge))
    for i = 1 : length(bandIms)
        thisDenoise(bandIms(i), baseDir, modName, outDirNN, modelDir);
    end
end

function multiModelDenoise2(baseDir, nameOfModel, outDirNN, modelDir)

bandIndices{1} = 1:10;
bandIndices{2} = 11:20;
bandIndices{3} = 21:30;

for thisBand = 1:3
    modName = ['model_' num2str(thisBand) '_0/' nameOfModel];
    bandIms = bandIndices{thisBand};
    for i = 1 : length(bandIms)
        thisDenoise(bandIms(i), baseDir, modName, outDirNN, modelDir);
    end
end

function multiModelDenoise3(baseDir, nameOfModel, outDirNN, modelDir)

load testInds;
load borders;

bandIndices{1} = 1:10;
bandIndices{2} = 11:20;
bandIndices{3} = 21:30;

testIs{1} = testInds(1:10);
testIs{2} = testInds(11:20);
testIs{3} = testInds(21:30);

for thisBand1 = 1:3
    bandIms = bandIndices{thisBand};
    thisI = testIs{thisBand1};
    
    for thisBand2 = 1:8
        modName = ['model_' num2str(thisBand1) '_' num2str(thisBand2) '/' nameOfModel];
        lowEdge = borders(thisBand2); highEdge = borders(thisBand2+1);
        bandIms2 = bandIms((thisI>=lowEdge)&(thisI<highEdge))
        for i = 1 : length(bandIms2)
            thisDenoise(bandIms2(i), baseDir, modName, outDirNN, modelDir);
        end
    end
end


function singleModelDenoise(baseDir, nameOfModel, outDirNN, modelDir)

for i = 1 : 30
    thisDenoise(i, baseDir, nameOfModel, outDirNN, modelDir);
end


function thisDenoise(ImNum, baseDir, nameOfModel, outDirNN, modelDir)

load(fullfile(modelDir, nameOfModel));
outDirNN2 = [baseDir outDirNN];
mkdir(outDirNN2);
net = vl_simplenn_tidy(net);
net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);

outName = ['Test_Image_' num2str(ImNum) '.png'];
imName = [baseDir 'Test_Image_' num2str(ImNum) '.png'];

zRGB = im2double(imread(imName));
input = single(zRGB);
res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test','CuDNN', 'true');

output = input - res(end).x;
output = round(output*255);
output(output<0) = 0;
output(output>255) = 255;
output = uint8(output);
imwrite(output,[outDirNN2 outName]);

