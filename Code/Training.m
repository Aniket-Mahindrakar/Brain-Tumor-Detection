clear all;
clc;
close all;
seed = 50;
rng(seed);

load('..\Data\Trainset.mat');
data = datasample(data,3064);
label = data(:,end);
data = data(:,1:end-1);

%SVM
svmParams = templateSVM('KernelFunction','rbf', 'KernelScale', 'auto', ...
    'Standardize', 1);
svm_Mdl = fitcecoc(data, label, 'Learners', svmParams, 'Coding', 'onevsall');
svm_CVMdl = crossval(svm_Mdl);
svm_loss = kfoldLoss(svm_CVMdl)

%Binary Decision Tree
tree_Mdl = fitctree(data,label);
tree_CVMdl = crossval(tree_Mdl);
tree_loss = kfoldLoss(tree_CVMdl)

%KNN
knn_Mdl = fitcknn(data, label);
knn_CVMdl = crossval(knn_Mdl);
knn_loss = kfoldLoss(knn_CVMdl)

%{
idx = kmedoids(data,3);
t = AccMeasure(label, idx)
ct = crosstab(idx, label);
purity = sum(max(ct)))/3064
%}
