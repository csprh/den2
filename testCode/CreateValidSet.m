function CreateValidSet;

baseDir = '/mnt/storage/home/csprh/scratch/HAB/Huwei/';
outDirClean = '/mnt/storage/home/csprh/scratch/HAB/Huwei/Valid/Clean/';
outDirNoisy = '/mnt/storage/home/csprh/scratch/HAB/Huwei/Valid/Noisy/';
inds = [1 172 73
    2 20  123
    3 252 1757
    4 47  2232
    5 33  642
    6 54  1280
    7 65  827
    8 29  2719
    9 106 2863
    10 119 3200
    11 351 57
    12 404 3200
    13 406 2945
    14 362 270
    15 383 439
    16 366 697
    17 374 1043
    18 378 2645
    19 385 1900
    20 388 2410
    21 773 3184
    22 823 46
    23 877 136
    24 950 2925
    25 952 2478
    26 659 2210
    27 683 523
    28 674 706
    29 665 1892
    30 678 1176];


classesA{1} = 'Buildings/';
classesA{2} = 'Foliage/';
classesA{3} = 'Text/';

theseInds = inds(:,3);


for ii = 1:30
    thisInd = theseInds(ii);
    imName = ['Image_' num2str(thisInd) '.png'];
    thisImName = ['Test_Image_' num2str(ii) '.png'];
    
    NoisyImageName = getThisPath(baseDir,classesA,imName,'Noisy/');
    CleanImageName = getThisPath(baseDir,classesA,imName,'Clean/');
    
    NoisyImage = imread(NoisyImageName);
    NoisyImage = imcrop(NoisyImage, [1 1 1023 1023]);
    imwrite(NoisyImage,[outDirNoisy thisImName]);
    
    CleanImage = imread(CleanImageName);
    CleanImage = imcrop(CleanImage, [1 1 1023 1023]);
    imwrite(CleanImage,[outDirClean thisImName]);
end



function output = getThisPath(baseDir, classesA,imName,cleanNoisy)

for ii = 1: length(classesA)
    thisClass = classesA{ii};
    thisPath = dir ([baseDir thisClass cleanNoisy imName]);
    if length(thisPath) == 1
        output = [baseDir thisClass cleanNoisy imName];
    end
end
