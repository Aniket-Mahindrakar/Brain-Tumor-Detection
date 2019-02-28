clear all;
clc;
close all;

rootFolder = '../Data_img/';
%categories = {'1', '2', '3'};

imds = imageDatastore(rootFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);


layers = [
    imageInputLayer([512 512 1])  
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer 
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

opts = trainingOptions('sgdm', ...
    'MaxEpochs',15, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',imdsValidation,...
    'ValidationPatience',Inf,...
    'ExecutionEnvironment','cpu');

net = trainNetwork(imds,layers,opts);