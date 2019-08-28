clear;
close;
clc;

data = [];
label = [];
corrupt_images = [];
for k = 1:3064
    load(strcat('../Data/',num2str(k),'.mat'));
    img = cjdata.image;
    img = uint8(255*mat2gray(img));
    
    %seg_img = cjdata.tumorMask;
    %seg_img = img.*uint8(seg_img);
    
    s = size(img);
    if ((s(1) ~= 512) || (s(2) ~= 512))
        img = imresize(img, 2);
        img = imresize(img, 2);
    end
    %seg_img = edge(seg_img, 'Roberts');
    l = cjdata.label;
    %figure, imshow(seg_img);title('Segmented Tumor');

    % Extract features using DWT
    x = double(img);
    m = size(img,1);
    n = size(img,2);
    %signal1 = (rand(m,1));
    %winsize = floor(size(x,1));
    %winsize = int32(floor(size(x)));
    %wininc = int32(10);
    %J = int32(floor(log(size(x,1))/log(2)));
    %Features = getmswpfeat(signal,winsize,wininc,J,'matlab');

    %m = size(img,1);
    %signal = rand(m,1);
    signal1 = img(:,:);
    %Feat = getmswpfeat(signal,winsize,wininc,J,'matlab');
    %Features = getmswpfeat(signal,winsize,wininc,J,'matlab');

    [cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
    [cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
    [cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');

    DWT_feat = [cA3,cH3,cV3,cD3];
    G = pca(DWT_feat);
    %whos DWT_feat
    %whos G
    g = graycomatrix(G);
    stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
    Contrast = stats.Contrast;
    Correlation = stats.Correlation;
    Energy = stats.Energy;
    Homogeneity = stats.Homogeneity;
    Mean = mean2(G);
    Standard_Deviation = std2(G);
    Entropy = entropy(G);
    RMS = mean2(rms(G));
    %Skewness = skewness(img)
    Variance = mean2(var(double(G)));
    a = sum(double(G(:)));
    Smoothness = 1-(1/(1+a));
    Kurtosis = kurtosis(double(G(:)));
    Skewness = skewness(double(G(:)));
    % Inverse Difference Movement
    m = size(G,1);
    n = size(G,2);
    in_diff = 0;
    for i = 1:m
        for j = 1:n
            temp = G(i,j)./(1+(i-j).^2);
            in_diff = in_diff+temp;
        end
    end
    IDM = double(in_diff);
    feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM, l];
    data=[data;feat];
    k
end

save('../Data/Trainset.mat','data');