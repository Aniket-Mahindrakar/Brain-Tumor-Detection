clear all;
clc;
close all;

for k = 1:3064
    load(strcat('..\Data\',num2str(k),'.mat'));
    
    img = cjdata.image;
    img = uint8(255 * mat2gray(img));
    label = cjdata.label;
    
    if label == 1
        outfile = strcat('..\Data_img\1\',num2str(k),'.jpg');
    elseif label == 2
        outfile = strcat('..\Data_img\2\',num2str(k),'.jpg');
    elseif label == 3
        outfile = strcat('..\Data_img\3\',num2str(k),'.jpg');
    end
    
    s = size(img);
    if ((s(1) ~= 512) || (s(2) ~= 512))
        img = imresize(img, 2);
    end
    
    imwrite(img, outfile);
    k
end