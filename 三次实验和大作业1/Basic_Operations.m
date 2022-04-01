%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   Image Basic Operations
%       图像的点运算、代数运算和几何运算练习
%%-------------------------------------------------------------------------
%% 图像的点运算
clc, clear, close all
filename='Nogizaka_dark.jpg';
imSrc = imread(filename);
figure('Name','原始图片','NumberTitle','off'),imshow(imSrc);
% imwrite(imSrc,'对数变换前的原始图像.png')
[hei, wid, len] = size(imSrc);
imDst=zeros(hei,wid,len);
Temp=mat2gray(log(1+double(imSrc)));    %对数变换并归一化
imDst=im2uint8(Temp);   %数据转换回uint8
figure('Name','对数变换后的图片','NumberTitle','off'),imshow(imDst);
% imwrite(imDst,'对数变换后的图像.png')

%% 图像的代数运算
clc, clear, close all
filename1='Akihabara.jpg';
filename2='Ginza.jpg';
imSrc1 = imread(filename1);
imSrc2 = imread(filename2);
figure('Name','原始图片1','NumberTitle','off'),imshow(imSrc1);
figure('Name','原始图片2','NumberTitle','off'),imshow(imSrc2);
% imwrite(imSrc1,'代数运算原始图片1.png')
% imwrite(imSrc2,'代数运算原始图片2.png')
[hei1, wid1, len1] = size(imSrc1);
[hei2, wid2, len2] = size(imSrc2);
hei=(hei1+hei2)/2; wid=(wid1+wid2)/2; len=(len1+len2)/2;
imDst=zeros(hei,wid,len);

figure('Name','代数运算结果图片','NumberTitle','off')
subplot(2,2,1)
Temp=mat2gray(double(imSrc1)+double(imSrc2));   %运算并归一化
imDst=im2uint8(Temp);   %数据转换回uint8
imshow(imDst)
title('加运算结果')
% imwrite(imDst,'加运算结果图片.png')
subplot(2,2,2)
Temp=mat2gray(abs(double(imSrc1)-double(imSrc2)));   %运算并归一化
imDst=im2uint8(Temp);   %数据转换回uint8
imshow(imDst)
title('减运算结果')
% imwrite(imDst,'减运算结果图片.png')
subplot(2,2,3)
Temp=mat2gray(double(imSrc1).*double(imSrc2));   %运算并归一化
imDst=im2uint8(Temp);   %数据转换回uint8
imshow(imDst)
title('乘运算结果')
% imwrite(imDst,'乘运算结果图片.png')
subplot(2,2,4)
Temp=mat2gray(double(imSrc1)./double(imSrc2)+ ...
    double(imSrc2)./double(imSrc1));   %运算并归一化
imDst=im2uint8(Temp);   %数据转换回uint8
imshow(imDst)
title('除运算结果')
% imwrite(imDst,'除运算结果图片.png')

%% 图像的几何运算
clc, clear, close all
filename='Fuji Television Building Small.jpg';
imSrc = imread(filename);
figure('Name','放缩操作的原始图片','NumberTitle','off'),imshow(imSrc);
% imwrite(imSrc,'放缩操作的原始图片.png')
[hei, wid, len] = size(imSrc);
imDst=zeros(hei,wid,len);
imDst=imresize(imSrc,1.5,'bilinear'); 
figure('Name','放大到1.5倍的图片','NumberTitle','off'),imshow(imDst);
% imwrite(imDst,'放大到1.5倍的图片.png')
imDst=imresize(imSrc,0.8,'bilinear'); 
figure('Name','缩小到0.8倍的图片','NumberTitle','off'),imshow(imDst);
% imwrite(imDst,'缩小到0.8倍的图片.png')

filename='Tokyo University Yasuda Koudou.jpg';
imSrc = imread(filename);
figure('Name','旋转操作的原始图片','NumberTitle','off'),imshow(imSrc);
% imwrite(imSrc,'旋转操作的原始图片.png')
imDst=imrotate(imSrc,-45); 
figure('Name','顺时针旋转45°的图片','NumberTitle','off'),imshow(imDst);
% imwrite(imDst,'顺时针旋转45°的图片.png')
