clear all; clc; close all;
a=rng;
out=randn;
rng(a)
out=randn;

rootFolder = '../Data_img/';

imds = imageDatastore(rootFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.6);
[imdsValidation, imdsTest] = splitEachLabel(imdsValidation, 0.5);

pixelRange = [-4 4];
rotationRange = [0 90];        
imageAugmenter = imageDataAugmenter(...
            'RandRotation',rotationRange, ...
            'RandXReflection',true,...
            'RandYReflection',true,...
            'RandXTranslation',pixelRange,...
            'RandYTranslation',pixelRange);
imageSize = [128 128 1];
datasource = augmentedImageDatastore(imageSize,imdsTrain,...
            'DataAugmentation',imageAugmenter,...
            'OutputSizeMode','randcrop');
        
optimVars = [
    optimizableVariable('InitialLearnRate',[0.001 0.005],'Transform','log');
    optimizableVariable('GradientDecayFactor',[0.8 0.95], 'Transform','log');
    optimizableVariable('SquaredGradientDecayFactor',[0.85 1],'Transform','log');
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log');
    optimizableVariable('MaxEpochs',[16 32],'Type','integer')];

i = 1;
ObjFcn = makeObjFcn(imdsTrain,imdsValidation,datasource,imageSize,i);
BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxObj',30,...
    'MaxTime',8*60*60,...
    'IsObjectiveDeterministic',false,...
    'UseParallel',false);


bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
valError = savedStruct.valError

[YPredicted,probs] = classify(savedStruct.trainedNet,imdsTest);
YTest = imdsTest.Labels;
testError = 1 - mean(YPredicted == YTest)

NTest = numel(YTest);
testErrorSE = sqrt(testError*(1-testError)/NTest);
testError95CI = [testError - 1.96*testErrorSE, testError + 1.96*testErrorSE]

figure
[cmat,classNames] = confusionmat(YTest,YPredicted);
h = heatmap(classNames,classNames,cmat);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix');
