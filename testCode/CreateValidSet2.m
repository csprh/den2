function CreateValidSet;

baseDir = '/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI/Train/';
outDirClean = '/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI/Valid4/Clean/';
outDirNoisy = '/home/cosc/csprh/linux/HABCODE/scratch/HUAWEI/Valid4/Noisy/';
inds = [1 126 72
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


classesA{1} = 'Buildings/';
classesA{2} = 'Foliage/';
classesA{3} = 'Text/';

theseInds = inds(:,2);


for ii = 1:30
    thisInd = theseInds(ii);
    imName = ['Image_' num2str(thisInd) '.png'];
    thisImName = ['Test_Image_' num2str(ii) '.png'];
    
    NoisyImageName = getThisPath(baseDir,classesA,imName,'Noisy/');
    CleanImageName = getThisPath(baseDir,classesA,imName,'Clean/');
    
    NoisyImage = imread(NoisyImageName);
    NoisyImage = imcrop(NoisyImage, [1+1024 1 1023+1024 1023]);
    imwrite(NoisyImage,[outDirNoisy thisImName]);
    
    CleanImage = imread(CleanImageName);
    CleanImage = imcrop(CleanImage, [1+1024 1 1023+1024 1023]);
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
