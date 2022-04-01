%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��4��
% ˵����   Denoise for Raw Image Demosaic
%       ��BM3D�㷨��Demosaic_raw.m�����ȥ������DOP��AOPͼ�����ȥ��
%       �õ��ǹٷ���BM3D�㷨��http://www.cs.tut.fi/~foi/GCF-BM3D/
%%-------------------------------------------------------------------------
%% BM3D�㷨ȥ��
addpath(genpath('bm3d_matlab_package'))
clear,clc,close all;

tic; 
img_RGB = imread('rawȥ�����˺�����DOP.bmp');  
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
% imwrite(y_est,'rawȥ������ȥ����DOP.bmp');
% imwrite(y_est,'rawȥ������ȥ����DOP.jpg');
% imwrite(y_est,'BM3Dȥ��DOP.png');
% disp('�������')

% y_est_hsv = rgb2hsv(y_est);
% figure(2),imshow(y_est_hsv)
% imwrite(y_est_hsv,'rawȥ������ȥ����AOP.bmp');
% imwrite(y_est_hsv,'rawȥ������ȥ����AOP.jpg');
% imwrite(y_est_hsv,'BM3Dȥ��AOP.png');
% disp('�������')
