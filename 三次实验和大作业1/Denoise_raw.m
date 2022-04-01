%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   Denoise for Raw Image Demosaic
%       用BM3D算法对Demosaic_raw.m输出的去马赛克DOP、AOP图像进行去噪
%       用的是官方的BM3D算法：http://www.cs.tut.fi/~foi/GCF-BM3D/
%%-------------------------------------------------------------------------
%% BM3D算法去噪
addpath(genpath('bm3d_matlab_package'))
clear,clc,close all;

tic; 
img_RGB = imread('raw去马赛克含噪声DOP.bmp');  
[hei,wid,di] = size(img_RGB);
y_est = zeros(hei,wid,di);

figure(1),imshow(img_RGB)
% figure(1),imshow(rgb2hsv(img_RGB))

for k = 1:3
    switch k
        case 1
            img = img_RGB(:,:,1);
        case 2
            img = img_RGB(:,:,2);
        case 3
            img = img_RGB(:,:,3);
    end
    
    if length(size(img)) > 2
        img = rgb2gray(img);
    end
    
    % figure(3),imshow(img);
    y = im2double(img);
    sigma=200/255;
    z = y;
    
    y_est(:,:,k) = BM3D(z,sigma);
    % figure(4),imshow(y_est);
    toc;
end

figure(2),imshow(y_est)
% imwrite(y_est,'raw去马赛克去噪声DOP.bmp');
% imwrite(y_est,'raw去马赛克去噪声DOP.jpg');
% imwrite(y_est,'BM3D去噪DOP.png');
% disp('保存完毕')

% y_est_hsv = rgb2hsv(y_est);
% figure(2),imshow(y_est_hsv)
% imwrite(y_est_hsv,'raw去马赛克去噪声AOP.bmp');
% imwrite(y_est_hsv,'raw去马赛克去噪声AOP.jpg');
% imwrite(y_est_hsv,'BM3D去噪AOP.png');
% disp('保存完毕')
