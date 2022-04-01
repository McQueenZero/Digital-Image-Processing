%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   图像边缘检测
%%-------------------------------------------------------------------------
%% 图像边缘检测（冈萨雷斯.MATLAB版数字图像处理,P.289）
clc, clear, close all
filename = 'building.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
% Sobel边缘检测器
[gS, tS] = edge(imSrc, 'sobel');
% Prewitt边缘检测器
[gP, tP] = edge(imSrc, 'prewitt');
% Roberts边缘检测器
[gR, tR] = edge(imSrc, 'roberts');
% LoG边缘检测器
[gL, tL] = edge(imSrc, 'log');
% Zero crossings边缘检测器
[gZ, tZ] = edge(imSrc, 'zerocross');
% Canny边缘检测器
[gC, tC] = edge(imSrc, 'canny');
% 输出
figure('Name','原始图片','NumberTitle','off')
imshow(imSrc)
figure('Name','不同边缘检测器处理结果','NumberTitle','off')
subplot(2,3,1)
imshow(gS),title('Sobel边缘检测器处理图片')
subplot(2,3,2)
imshow(gP),title('Prewitt边缘检测器处理图片')
subplot(2,3,3)
imshow(gR),title('Roberts边缘检测器处理图片')
subplot(2,3,4)
imshow(gL),title('LoG边缘检测器处理图片')
subplot(2,3,5)
imshow(gZ),title('Zero crossings边缘检测器处理图片')
subplot(2,3,6)
imshow(gC),title('Canny边缘检测器处理图片')

