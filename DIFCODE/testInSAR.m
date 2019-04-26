windowSize = 128;sigma = 20; multFact = 40000; fKappa = 20;
g = fspecial('gaussian', windowSize, sigma);

gMult = g*multFact;

cmplxG = exp(i*gMult);
inSARTest = angle(cmplxG);

subplot(1,6,1);
imagesc(inSARTest);
title ('Noise Free inSAR');
inSARTestFlat = inSARTest(:);
[tfVMVariates] = vmrand(inSARTestFlat, ones(length(inSARTestFlat),1)*fKappa);
inSARTestNoise = reshape(tfVMVariates,size(inSARTest));
subplot(1,6,2);

imagesc(inSARTestNoise);
title ('Noisy inSAR');
cmplxG2 = exp(i*inSARTestNoise*2);
inSARTest2 = angle(cmplxG2);
subplot(1,6,3);

imagesc(inSARTest2);
title ('Wrapped*2');

cmplxG3 = exp(i*inSARTestNoise*3);
inSARTest3 = angle(cmplxG3);
subplot(1,6,4);

imagesc(inSARTest3);
title ('Wrapped*3');
cmplxG4 = exp(i*inSARTestNoise*4);
inSARTest4 = angle(cmplxG4);

subplot(1,6,5);

imagesc(inSARTest4);
title ('Wrapped*4');

cmplxG5 = exp(i*inSARTestNoise*5);
inSARTest5 = angle(cmplxG5);

subplot(1,6,6);

imagesc(inSARTest5);
title ('Wrapped*5');

