%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   图像小波变换
%%-------------------------------------------------------------------------
% Reference: 冈萨雷斯.数字图像处理,P.290小波和多分辨率处理
%% 提取图像变换特征（冈萨雷斯.MATLAB版数字图像处理,P.184小波变换）
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
imSrc = rgb2gray(imSrc);
[hei, wid, dim] = size(imSrc);
% 二维单尺度Haar小波分解
[cA1, cH1, cV1, cD1] = dwt2(imSrc, 'haar');
figure('Name','原始图片','NumberTitle','off')
imshow(imSrc)
figure('Name','二维单尺度Haar小波分解图','NumberTitle','off')
subplot(2,2,1), imshow(cA1,[]),title('图像的近似（低频部分）')
subplot(2,2,2), imshow(cH1,[]),title('图像的轮廓（水平）')
subplot(2,2,3), imshow(cV1,[]),title('图像的轮廓（垂直）')
subplot(2,2,4), imshow(cD1,[]),title('图像的轮廓（对角）')
% 用upcoef2函数从系数中重构近似和细节
A1 = upcoef2('a', cA1, 'bior3.7', 1);
H1 = upcoef2('d', cH1, 'bior3.7', 1);
V1 = upcoef2('v', cV1, 'bior3.7', 1);
D1 = upcoef2('d', cD1, 'bior3.7', 1);
figure('Name','upcoef2函数重构的近似细节图','NumberTitle','off')
subplot(2,2,1), imshow(A1, []),title('图像的近似（低频部分）')
subplot(2,2,2), imshow(H1, []),title('图像的轮廓（水平）')
subplot(2,2,3), imshow(V1, []),title('图像的轮廓（垂直）')
subplot(2,2,4), imshow(D1, []),title('图像的轮廓（对角）')
% 利用waverec2函数进行小波重构
[C, S] = wavedec2(imSrc, 2, 'bior3.7');
I_wrec = waverec2(C, S, 'bior3.7');
figure('Name','waverec2函数重构结果图','NumberTitle','off')
imshow(I_wrec, [])
% 利用appcoef2函数抽取第一层低频近似系数和第二层低频近似系数
wcA1 = appcoef2(C, S, 'bior3.7', 1);
wcA2 = appcoef2(C, S, 'bior3.7', 2);
figure('Name','appcoef2函数抽取的第一层和第二层近似系数','NumberTitle','off')
subplot(1,2,1),imshow(wcA1, []),title('第一层近似系数')
subplot(1,2,2),imshow(wcA2, []),title('第二层近似系数')
% 使用upcoef函数重构
I_wup1 = upcoef2('a', wcA1, 'bior3.7', 1);
I_wup2 = upcoef2('a', wcA2, 'bior3.7', 2);
figure('Name','upcoef函数重构的第一层和第二层近似结果','NumberTitle','off')
subplot(1,2,1),imshow(I_wup1, []),title('第一层重构近似结果')
subplot(1,2,2),imshow(I_wup2, []),title('第二层重构近似结果')
% 使用detcoef2函数抽取第一层细节
[chd1, cvd1, cdd1] = detcoef2('all', C, S, 1);
figure('Name','使用detcoef2函数抽取的第一层细节','NumberTitle','off')
subplot(1,3,1), imshow(chd1, []),title('图像的轮廓（水平）')
subplot(1,3,2), imshow(cvd1, []),title('图像的轮廓（垂直）')
subplot(1,3,3), imshow(cdd1, []),title('图像的轮廓（对角）')

