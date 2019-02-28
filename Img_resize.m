clear all;
clc;
files = dir('/Users/SriKrishnaManoj/Brain-Tumor-Detection/Data_img/2/*.jpg') ;    % you are in folder of csv files
for file = files'
    file.name
    I = imread(file.name);
    I = imresize(I, [128 128]);
    imwrite(I, file.name);
end
