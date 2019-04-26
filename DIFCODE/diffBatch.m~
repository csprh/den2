function diffBatch;

addpath('../BM3D');
addpath('../LASIP');
addpath('../DNCNN/model/specifics');
addpath('../DNCNN/utilities');

if ismac
    baseDir = '/Users/csprh/Dlaptop/MATLAB/MYCODE/DENOISE/den2/data/';

    run /Users/csprh/Dlaptop/MATLAB/TOOLBOXES/matconvnet-1.0-beta25/matlab/vl_setupnn
else
    baseDir = '/mnt/storage/home/csprh/scratch/HAB/Huwei/Train/';
    run /home/cosc/csprh/linux/code/matconvnet-1.0-beta25/matlab/vl_setupnn

end
noiseSigma = 25;


noisyDir = [baseDir 'AWGN25/Train/'];
cleanDir = [baseDir 'Origs/Train/'];
createNoiseAndDenoiseIms(cleanDir, noisyDir, noiseSigma);

noisyDir = [baseDir 'AWGN25/Test/Set68/'];
cleanDir = [baseDir 'Origs/Test/Set68/'];
createNoiseAndDenoiseIms(cleanDir, noisyDir, noiseSigma);

noisyDir = [baseDir 'AWGN25/Test/Set12/'];
cleanDir = [baseDir 'Origs/Test/Set12/'];
createNoiseAndDenoiseIms(cleanDir, noisyDir, noiseSigma);


function createNoiseAndDenoiseIms(cleanDir, noisyDir, noiseSigma )
modelSigma  = min(75,max(10,round(noiseSigma/5)*5)); %%% model noise level
load(['sigma=',num2str(modelSigma,'%02d'),'.mat']);

noiseDir = [noisyDir 'NoiseOrigs'];
denoiseDir0 = [noisyDir 'Denoise0'];
denoiseDir1 = [noisyDir 'Denoise1'];
%%%
net = vl_simplenn_tidy(net);

imList = dir([ cleanDir '*.png']);
noOfIms = length(imList);
for ii = 1:noOfIms
    thisImName = imList(ii).name;
    clean = im2double(imread([cleanDir thisImName]));
    %clean = single(clean);
    randn('seed',0);
    input = single(clean + noiseSigma/255*randn(size(clean)));

    imwrite(input, [noiseDir '/' thisImName]);
    input = single((double(imread([noiseDir '/' thisImName]))./255));
    res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.
    output0 = input - res(end).x;
    imwrite(output0 , [denoiseDir0 '/' thisImName]);
    [PSNRNoise, ~] = Cal_PSNRSSIM(255*clean,255*input,0,0);
    [PSNRDenoise0, ~] = Cal_PSNRSSIM(255*clean,255*output0,0,0);
    %[PSNRNN(ii), SSIMNN(ii)] = Cal_PSNRSSIM(255*clean,255*input,0,0);
    
    [PSNRDenoise1, output1] = BM3D(clean, input, noiseSigma,'high',0);
    [PSNRDenoise2, ~] = Cal_PSNRSSIM(255*clean,255*output1,0,0);
    imwrite(output1 , [denoiseDir1 '/' thisImName]);

    pause(0.1);
end
