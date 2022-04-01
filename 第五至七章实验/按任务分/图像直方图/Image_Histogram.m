%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   图像直方图
% 注意：   分小节运行
%%-------------------------------------------------------------------------
% 直方图均衡化：把原始图的直方图变换为均匀分布的形式，
% 增加像素灰度值的动态范围，提高图像对比度
%%-------------------------------------------------------------------------
%% 灰度图像直方图
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imGray = imSrc;
imGrayhisteq = histeq(imGray,256);   %直方图均衡化
figure('Name','直方图','NumberTitle','off')
subplot(1,2,1),imhist(imGray),title('直方图');    %直方图
subplot(1,2,2),imhist(imGrayhisteq),title('均衡化处理过的直方图');
figure('Name','灰度图像','NumberTitle','off')
subplot(1,2,1),imshow(imGray),title('灰度图片');
subplot(1,2,2),imshow(imGrayhisteq),title('均衡化处理过的灰度图');

%% 彩色图像直方图
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imRGBhisteq = histeq(imSrc,256);   %直方图均衡化
figure('Name','直方图','NumberTitle','off')
subplot(1,2,1),imhist(imSrc),title('直方图');    %直方图
subplot(1,2,2),imhist(imRGBhisteq),title('均衡化处理过的直方图');
figure('Name','彩色图像','NumberTitle','off')
subplot(1,2,1),imshow(imSrc),title('彩色图片');
subplot(1,2,2),imshow(imRGBhisteq),title('均衡化处理过的彩色图');

