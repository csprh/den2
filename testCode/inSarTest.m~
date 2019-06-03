function inSarTest;
gpuDevice();

run /home/cosc/csprh/linux/code/matconvnet-1.0-beta25/matlab/vl_setupnn


scratchDir = '/space/csprh/inSAR';
inputDir  = [scratchDir '/outgrad/'];
deformDir  = [scratchDir '/grad/'];
wrappedDir  = [scratchDir '/out/'];
testDir = [scratchDir '/test/'];
hatDir = [scratchDir '/deformHat/'];

modName = '/home/cosc/csprh/linux/code/den2/DnCNN_inSAR/TrainingCodes/DnCNN_InSAR/data/model_InSAR_1_0/model_InSAR_1_0-epoch-70.mat';

thisInd = 0;
for ii = 350 : 350+50
    outPSNR(ii) = thisDenoise(ii, deformDir, inputDir, wrappedDir, hatDir,  testDir, modName);
end
outPSNRMean = mean(outPSNR);

function PSNR = thisDenoise(ImNum, deformDir, inputDir,wrappedDir, hatDir, testDir,  nameOfModel)

load(fullfile(nameOfModel));

net = vl_simplenn_tidy(net);
net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);

numStrOutImGrad = sprintf([inputDir 'OOO%05d.png'],ImNum);
numStrOutDeform = sprintf([deformDir 'D%05d.png'],ImNum);
numStrOutWrappedIn = sprintf([wrappedDir 'O%05d.png'],ImNum); 
outStr = sprintf([hatDir 'H%05d.png'],ImNum);

zRGB = im2double(imread(numStrOutImGrad));
yRGB = im2double(imread(numStrOutDeform));
wRGB = im2double(imread(numStrOutWrappedIn));
input = single(zRGB);
clean = single(yRGB);

res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test','CuDNN', 'true');

outIm = input - res(end).x;

scaledgx = double(outIm(:,:,1));
scaledgy = double(outIm(:,:,2));

scaledgx = ((scaledgx/255.0)*(2*pi))-pi;
scaledgy = ((scaledgy/255.0)*(2*pi))-pi;

wrappedPhaseIn = ((wRGB/255.0)*(2*pi))-pi; 

phat = intgrad2(scaledgx,scaledgy,[],[],wrappedPhaseIn(0));
phat_wrapped = angle(exp(1i*phat));

outImWrap = round(255*(phat_wrapped+pi)/(2*pi));
imwrite(uint8(outImWrap), outStr);

refscaledgx = double(clean(:,:,1));
refscaledgy = double(clean(:,:,2));

refscaledgx = ((refscaledgx/255.0)*(2*pi))-pi;
refscaledgy = ((refscaledgy/255.0)*(2*pi))-pi;
PSNR = 0;
%[PSNR, SSIM] = Cal_PSNRSSIM(255*clean,255*outIm,0,0);


