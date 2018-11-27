function GenerateTrainingPatches_Banded

%%% Generate the training data.

clear;close all;


%Data size should approximate
%212096         128        1657 (from bw orig)
%40          40           1      212096
%40          40           3      488192
addpath('utilities');

batchSize      = 128;        %%% batch size


if ismac
    baseDir = '/Users/csprh/tmp/Huawei/Train/';
else
    baseDir = '/mnt/storage/home/csprh/scratch/HAB/Huwei/Train/';
end




for theseClasses = 0:3
    for theseBands = 0:8
        createPatches(theseClasses,theseBands, baseDir,  batchSize);
    end
end



function createPatches(theseClasses,theseBands, baseDir, batchSize)

load theseIndices

dataName      = '/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI/TrainingPatches';
patchsize     = 40;

step          = 0;

count   = 0;
classesA{1} = 'Buildings/';
classesA{2} = 'Foliage/';
classesA{3} = 'Text/';

if (theseClasses==0)
    classesB = classesA;
    if (theseBands==0)
        theseNHO = inds.NHO;
    else
        theseNHO = indsAll{theseBands}.NHO;
    end
else
    classesB = {classesA{theseClasses}};
    if (theseBands==0)
        theseNHO = indsCls{theseClasses}.NHO;
    else
        theseNHO = indsALLCls{theseBands}{theseClasses}.NHO;
    end
end

numberOfIms = length(theseNHO);

prop1 = numberOfIms/781;
stride        = round(240*sqrt(prop1)); 

for ii = 1:length(theseNHO)
    %for ii = 1:5
    thisInd = theseNHO(ii);
    %thisInd = 663;
    imName = ['Image_' num2str(thisInd) '.png'];
    filepaths{ii}.Clean = getThisPath(baseDir,classesB,imName,'Clean/');
    filepaths{ii}.Noisy = getThisPath(baseDir,classesB,imName,'Noisy/');
end


%% count the number of extracted patches
for i = 1 : length(filepaths)
    
    image = imread(filepaths{i}.Clean); % uint8
    
    %[~, name, exte] = fileparts(filepaths(i).name);
    if mod(i,100)==0
        disp([i,length(filepaths)]);
    end
    for s = 1:1
        %image = imresize(image,scales(s),'bicubic');
        [hei,wid,~] = size(image);
        for x = 1+step : stride : (hei-patchsize+1)
            for y = 1+step :stride : (wid-patchsize+1)
                count = count+1;
            end
        end
    end
end

numPatches = ceil(count/batchSize)*batchSize;

disp([numPatches,batchSize,numPatches/batchSize]);

%pause;

inputs  = zeros(patchsize, patchsize, 3, numPatches,'single'); % this is fast
cleaninputs  = zeros(patchsize, patchsize, 3, numPatches,'single'); % this is fast
count   = 0;
tic;
for i = 1 : length(filepaths)
    
    image = imread(filepaths{i}.Noisy); % uint8
    cleanImage = imread(filepaths{i}.Clean); % uint8
    
    if mod(i,100)==0
        disp([i,length(filepaths)]);
    end
    %     end
    %image_clip = image(1:180,1:180,:);
    %im_label  = im2single(image_clip); % single
    
    im_label  = im2single(image); % single
    cleanim_label  = im2single(cleanImage); % single
    [hei,wid,~] = size(im_label);
    
    for x = 1+step : stride : (hei-patchsize+1)
        for y = 1+step :stride : (wid-patchsize+1)
            count       = count+1;
            inputs(:, :, :, count)   = im_label(x : x+patchsize-1, y : y+patchsize-1,:);
            cleaninputs(:, :, :, count)   = cleanim_label(x : x+patchsize-1, y : y+patchsize-1,:);
        end
    end
end
toc;
set    = uint8(ones(1,size(inputs,4)));

disp('-------Datasize-------')
disp([size(inputs,4),batchSize,size(inputs,4)/batchSize]);

if ~exist(dataName,'file')
    mkdir(dataName);
end

%%% save data
save(fullfile(dataName,['imdb_',num2str(theseClasses),'_',num2str(theseBands)]), 'inputs','cleaninputs','set','-v7.3')

function output = getThisPath(baseDir, classesA,imName,cleanNoisy)

for ii = 1: length(classesA)
    thisClass = classesA{ii};
    thisPath = dir ([baseDir thisClass cleanNoisy imName]);
    if length(thisPath) == 1
        output = [baseDir thisClass cleanNoisy imName];
    end
end



