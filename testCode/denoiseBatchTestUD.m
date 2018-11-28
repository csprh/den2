function CropIms;

addpath('../LASIP');
addpath('../DNCNN/model');
addpath('../DNCNN/utilities');
run /home/cosc/csprh/linux/code/matconvnet-1.0-beta25/matlab/vl_setupnn

baseDir = '/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI/Test/';
%load(fullfile('/mnt/storage/home/csprh/code/HUAWEI/DNCNN/TrainingCodes/DnCNNHuawei/data/model_Huwei_All/model_Huwei_All-epoch-19.mat'));
load(fullfile('/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI/Models/model_0_0/out-epoch-60.mat'));
load(fullfile('/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI/Models/model_Huwei_All50-epoch-27.mat'));
net = vl_simplenn_tidy(net);
net.layers = net.layers(1:end-1);
net = vl_simplenn_tidy(net);



outDirDnCNN = [ baseDir 'OutUoBCNN_2/'];
mkdir(outDirDnCNN);


for i = 1 : 30
    outName = ['Test_Image_' num2str(i) '.png'];
    imName = [baseDir 'Test_Image_' num2str(i) '.png'];
    
    zRGB = im2double(imread(imName)); % uint8
    input = single(zRGB);
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test','CuDNN', 'false');
    %res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.
    output = input - res(end).x;
    output = round(output*255);
    output(output<0) = 0;
    output(output>255) = 255;
    output = uint8(output);
    imwrite(output,[outDirDnCNN outName]);
end


