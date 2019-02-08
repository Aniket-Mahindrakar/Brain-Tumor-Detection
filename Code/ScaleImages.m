clear;
close all;
clc;

for i = 1:3064
    load(strcat(num2str(i),'.mat'));
    img = cjdata.image;
    cjdata.image = uint8(255 * mat2gray(img));
    save(strcat(num2str(i),'.mat'),'cjdata');
end