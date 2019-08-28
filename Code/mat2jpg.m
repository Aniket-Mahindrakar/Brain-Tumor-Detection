clear all;
clc;
close all;

for k = 1:3064
    load(strcat('../Data/',num2str(k),'.mat'));
    
    img = cjdata.image;
    img = double(mat2gray(img));
    label = cjdata.label;
    
    if label == 1
        outfile = strcat('..\Data_img\1\',num2str(k),'.jpg');
    elseif label == 2
        outfile = strcat('..\Data_img\2\',num2str(k),'.jpg');
    elseif label == 3
        outfile = strcat('..\Data_img\3\',num2str(k),'.jpg');
    end
    
    img = imresize(img, [128, 128]);
    
    imwrite(img, outfile);
    k
end