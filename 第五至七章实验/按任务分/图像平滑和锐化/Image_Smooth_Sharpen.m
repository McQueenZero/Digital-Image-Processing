%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   图像平滑和锐化
% 注意：   分小节运行
%%-------------------------------------------------------------------------
% 平滑：空域低通滤波器
% 锐化：空域高通滤波器
% 空域滤波器：模板、卷积的原理
% 中值滤波：像素按灰度值大小排序，中心点取排序位置居中的像素灰度值
% 拉普拉斯算子：计算导数（梯度）或有限差分实现
%%-------------------------------------------------------------------------
%% 空域滤波图像平滑
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imNosSP=imnoise(imSrc,'salt & pepper',0.02);
imNosGAS=imnoise(imSrc,'gaussian');
figure('Name','原始图像及噪声图像','NumberTitle','off')
subplot(1,3,1),imshow(imSrc),title('原始图片');
subplot(1,3,2),imshow(imNosSP),title('椒盐噪声图片');
subplot(1,3,3),imshow(imNosGAS),title('高斯噪声图片');

imDstSP3x3=medfilt2(imNosSP,[3 3]);
imDstSP5x5=medfilt2(imNosSP,[5 5]);
imDstGAS3x3=medfilt2(imNosSP,[3 3]);
imDstGAS5x5=medfilt2(imNosSP,[5 5]);
figure('Name','中值滤波去噪对比','NumberTitle','off')
subplot(2,2,1),imshow(imDstSP3x3),title('空域3x3中值滤波去椒盐噪声图片');
subplot(2,2,2),imshow(imDstSP5x5),title('空域5x5中值滤波去椒盐噪声图片');
subplot(2,2,3),imshow(imDstGAS3x3),title('空域3x3中值滤波去高斯噪声图片');
subplot(2,2,4),imshow(imDstGAS5x5),title('空域5x5中值滤波去高斯噪声图片');

%% 拉普拉斯算子图像锐化
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
m4 = fspecial('laplacian',0);   %中心为-4的拉普拉斯滤波器
m8 = [1 1 1; 1 -8 1; 1 1 1];    %中心为-8的拉普拉斯滤波器
imf = im2double(imSrc);
imDstm4 = imf-imfilter(imf,m4,'symmetric');
imDstm8 = imf-imfilter(imf,m8,'symmetric');
figure('Name','图像锐化','NumberTitle','off')
subplot(1,3,1),imshow(imSrc),title('原始图片');
subplot(1,3,2),imshow(imDstm4),title('中心为-4的拉普拉斯滤波器处理所得图片');
subplot(1,3,3),imshow(imDstm8),title('中心为-8的拉普拉斯滤波器处理所得图片');
