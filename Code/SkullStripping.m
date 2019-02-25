clear all;
clc;
close all;

imageNo = 18;
load(strcat('../Data/',num2str(imageNo),'.mat'));

I = uint8(255 * mat2gray(cjdata.image));
% Get the dimensions of the image.
% numberOfColorBands should be = 1.
[rows, columns, numberOfColorBands] = size(I);
if numberOfColorBands > 1
  % It's not really gray scale like we expected - it's color.
  % Convert it to gray scale by taking only the green channel.
  I = I(:, :, 2); % Take green channel.
end
% Display the original gray scale image.
subplot(2, 3, 1);
imshow(I, []);
axis on;
title('Original Grayscale Image');
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Give a name to the title bar.
set(gcf, 'Name', 'Demo by ImageAnalyst', 'NumberTitle', 'Off')
% Let's compute and display the histogram.
[pixelCount, grayLevels] = imhist(I);
subplot(2, 3, 2);
bar(grayLevels, pixelCount);
grid on;
title('Histogram of original image');
xlim([0 grayLevels(end)]); % Scale x axis manually.
% Crop image to get rid of light box surrounding the image
I = I(3:end-3, 4:end-4);
% Threshold to create a binary image
binaryImage = I > 20;
% Get rid of small specks of noise
binaryImage = bwareaopen(binaryImage, 10);
% Display the original gray scale image.
subplot(2, 3, 3);
imshow(binaryImage, []);
axis on;
title('Binary Image');
% Seal off the bottom of the head - make the last row white.
binaryImage(end,:) = true;
% Fill the image
binaryImage = imfill(binaryImage, 'holes');
subplot(2, 3, 4);
imshow(binaryImage, []);
axis on;
title('Cleaned Binary Image');
% Erode away 15 layers of pixels.
se = strel('disk', 15, 0);
binaryImage = imerode(binaryImage, se);
subplot(2, 3, 5);
imshow(binaryImage, []);
axis on;
title('Eroded Binary Image');
% Mask the gray image
finalImage = I; % Initialize.
finalImage(~binaryImage) = 0;
subplot(2, 3, 6);
imshow(finalImage, []);
axis on;
title('Skull stripped Image');

% K means Clustering to segment tumor
cform = makecform('srgb2lab');
% Apply the colorform
cmap = jet ;
Irgb = ind2rgb(finalImage, cmap);
lab_he = applycform(Irgb,cform);

% Classify the colors in a*b* colorspace using K means clustering.
% Since the image has 3 colors create 3 clusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 1;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',1);
%[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% Label every pixel in tha image using results from K means
pixel_labels = reshape(cluster_idx,nrows,ncols);
%figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');

% Create a blank cell array to store the results of clustering
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = finalImage;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end

%
figure, subplot(2,2,1), imshow(segmented_images{1});title('Objects in Cluster 1');
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);

%figure, imshow(segmented_images{2});title('Objects in Cluster 2');

seg_img = im2bw(segmented_images{1});
subplot(2,2,2), imshow(seg_img);title('Segmented Tumor');

se = strel('disk', 12);
img_open = imopen(seg_img, se);
subplot(2,2,3), imshowpair(I, img_open), title('Morphological Segmented Image');
subplot(2,2,4), imshowpair(I, cjdata.tumorMask), title('Original Segmented Image');